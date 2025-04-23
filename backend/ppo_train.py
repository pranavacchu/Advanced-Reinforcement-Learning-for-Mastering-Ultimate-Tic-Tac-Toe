import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from torch.distributions import Categorical
import os
from tqdm import tqdm
import wandb
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR
import gc
import time
import math
from concurrent.futures import ThreadPoolExecutor
import torch.multiprocessing as mp

# --- Game Constants and Logic ---
BOARD_SIZE = 9
SUBBOARD_SIZE = 3
EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1

# --- Optimized Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = torch.cuda.is_available()
scaler = GradScaler(enabled=use_amp)
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Parallel Environment Processing ---
class ParallelGameState:
    def __init__(self, num_envs=4):
        self.num_envs = num_envs
        self.states = [initialize_game_state() for _ in range(num_envs)]
        self.pool = ThreadPoolExecutor(max_workers=num_envs)
    
    def step(self, actions):
        futures = []
        for i in range(self.num_envs):
            if actions[i] is not None:
                row, col = actions[i] // BOARD_SIZE, actions[i] % BOARD_SIZE
                futures.append(self.pool.submit(make_move, self.states[i], row, col))
            else:
                futures.append(self.pool.submit(initialize_game_state))
        
        self.states = [f.result() for f in futures]
        return self.states
    
    def reset(self):
        self.states = [initialize_game_state() for _ in range(self.num_envs)]
        return self.states

# --- Vectorized Game Functions ---
def check_sub_board_winner_vectorized(board, sub_row, sub_col):
    start_row = sub_row * 3
    start_col = sub_col * 3
    sub_board = board[start_row:start_row+3, start_col:start_col+3]
    
    # Check rows
    row_wins = torch.any(torch.all(sub_board == 1, dim=1)) or torch.any(torch.all(sub_board == -1, dim=1))
    # Check columns
    col_wins = torch.any(torch.all(sub_board == 1, dim=0)) or torch.any(torch.all(sub_board == -1, dim=0))
    # Check diagonals
    diag1 = torch.all(torch.diag(sub_board) == 1) or torch.all(torch.diag(sub_board) == -1)
    diag2 = torch.all(torch.diag(torch.flip(sub_board, [0])) == 1) or torch.all(torch.diag(torch.flip(sub_board, [0])) == -1)
    
    if row_wins or col_wins or diag1 or diag2:
        return 1 if torch.any(sub_board == 1) else -1
    return 0

def get_valid_moves_vectorized(game_state):
    board = torch.tensor(game_state["board"], device=device)
    active_row, active_col = game_state["active_sub_row"], game_state["active_sub_col"]
    
    if active_row is not None and active_col is not None:
        start_row, start_col = active_row * 3, active_col * 3
        valid_mask = (board[start_row:start_row+3, start_col:start_col+3] == EMPTY)
        if torch.any(valid_mask):
            valid_moves = [(start_row + r, start_col + c) for r, c in torch.nonzero(valid_mask)]
            return valid_moves
    
    valid_moves = []
    for sub_r in range(SUBBOARD_SIZE):
        for sub_c in range(SUBBOARD_SIZE):
            if is_sub_board_playable(game_state, sub_r, sub_c):
                start_row, start_col = sub_r * 3, sub_c * 3
                sub_valid = (board[start_row:start_row+3, start_col:start_col+3] == EMPTY)
                moves = [(start_row + r, start_col + c) for r, c in torch.nonzero(sub_valid)]
                valid_moves.extend(moves)
    return valid_moves

# --- Network Components ---
class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
        
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out

# --- Memory Management ---
class PPOMemory:
    def __init__(self, batch_size, max_size, state_shape, device):
        try:
            self.batch_size = batch_size
            self.max_size = max_size
            self.device = device
            
            # Pre-allocate memory
            self.states = torch.zeros((max_size, *state_shape), dtype=torch.float32, device=device)
            self.actions = torch.zeros(max_size, dtype=torch.long, device=device)
            self.log_probs = torch.zeros(max_size, dtype=torch.float32, device=device)
            self.values = torch.zeros(max_size, dtype=torch.float32, device=device)
            self.rewards = torch.zeros(max_size, dtype=torch.float32, device=device)
            self.dones = torch.zeros(max_size, dtype=torch.bool, device=device)
            
            self.ptr = 0
            self.size = 0
        except Exception as e:
            print(f"Error initializing PPOMemory: {e}")
            raise
    
    def store_memory(self, state, action, log_prob, value, reward, done):
        try:
            idx = self.ptr % self.max_size
            self.states[idx] = state
            self.actions[idx] = action
            self.log_probs[idx] = log_prob
            self.values[idx] = value
            self.rewards[idx] = reward
            self.dones[idx] = done
            
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
        except Exception as e:
            print(f"Error storing memory: {e}")
            raise
    
    def generate_batches(self, advantages, returns):
        try:
            batch_states = []
            batch_actions = []
            batch_log_probs = []
            batch_old_values = []
            batch_advantages = []
            batch_returns = []
            
            indices = torch.randperm(self.size, device=self.device)
            
            for start_idx in range(0, self.size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, self.size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states.append(self.states[batch_indices])
                batch_actions.append(self.actions[batch_indices])
                batch_log_probs.append(self.log_probs[batch_indices])
                batch_old_values.append(self.values[batch_indices])
                batch_advantages.append(advantages[batch_indices])
                batch_returns.append(returns[batch_indices])
            
            return zip(batch_states, batch_actions, batch_log_probs, 
                      batch_old_values, batch_advantages, batch_returns)
        except Exception as e:
            print(f"Error generating batches: {e}")
            raise
    
    def clear_memory(self):
        try:
            self.ptr = 0
            self.size = 0
        except Exception as e:
            print(f"Error clearing memory: {e}")
            raise

# --- Optimized PPO Network ---
class PPONetwork(nn.Module):
    def __init__(self, input_channels=5, num_res_blocks=4, num_filters=128):
        super(PPONetwork, self).__init__()
        self.initial_conv = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_filters)
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_filters) for _ in range(num_res_blocks)])
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * BOARD_SIZE * BOARD_SIZE, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Ensure input has correct dimensions [batch_size, channels, height, width]
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        x = self.res_blocks(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

# --- Optimized PPO Agent ---
class PPOAgent:
    def __init__(self, state_shape, n_actions, lr=3e-4, n_steps=1024, batch_size=256,
                 n_epochs=4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, total_updates=10000,
                 clip_vloss=True, num_envs=4):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.clip_vloss = clip_vloss
        self.device = device
        self.num_envs = num_envs
        
        self.policy = PPONetwork(input_channels=state_shape[0]).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_updates)
        self.memory = PPOMemory(self.batch_size, self.n_steps * num_envs, state_shape, device)
        self.best_avg_reward = -float('inf')
        
        # Initialize parallel environments
        self.envs = ParallelGameState(num_envs=num_envs)

    def get_state_tensor(self, game_states):
        batch_size = len(game_states)
        board_tensor = torch.zeros((batch_size, 5, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32, device=device)
        
        for i, game_state in enumerate(game_states):
            # Channel 0: X pieces (Player 1)
            # Channel 1: O pieces (Player -1)
            # Channel 2: Meta board winners
            # Channel 3: Active sub-board indicator
            # Channel 4: Current player indicator
            
            board = torch.tensor(game_state["board"], device=device)
            board_tensor[i, 0] = (board == PLAYER_X).float()
            board_tensor[i, 1] = (board == PLAYER_O).float()
            
            meta_board = torch.tensor(game_state["meta_board"], device=device)
            for r in range(SUBBOARD_SIZE):
                for c in range(SUBBOARD_SIZE):
                    if meta_board[r, c] != EMPTY:
                        board_tensor[i, 2, r*3:(r+1)*3, c*3:(c+1)*3] = meta_board[r, c]
            
            active_row, active_col = game_state["active_sub_row"], game_state["active_sub_col"]
            if active_row is not None and active_col is not None:
                if is_sub_board_playable(game_state, active_row, active_col):
                    start_row, start_col = active_row * 3, active_col * 3
                    board_tensor[i, 3, start_row:start_row+3, start_col:start_col+3] = 1.0
                else:
                    for sub_r in range(SUBBOARD_SIZE):
                        for sub_c in range(SUBBOARD_SIZE):
                            if is_sub_board_playable(game_state, sub_r, sub_c):
                                start_row, start_col = sub_r * 3, sub_c * 3
                                board_tensor[i, 3, start_row:start_row+3, start_col:start_col+3] = 1.0
            
            current_player = 1.0 if game_state["current_player"] == PLAYER_X else -1.0
            board_tensor[i, 4] = current_player
        
        return board_tensor

    def get_valid_actions(self, game_states):
        valid_actions = []
        for game_state in game_states:
            valid_moves = get_valid_moves_vectorized(game_state)
            valid_actions.append([r * 9 + c for r, c in valid_moves])
        return valid_actions

    def choose_action(self, game_states):
        states = self.get_state_tensor(game_states)
        with torch.no_grad():
            with autocast(device_type=device_type, enabled=use_amp):
                policy, value = self.policy(states)
        
        valid_actions = self.get_valid_actions(game_states)
        actions = []
        log_probs = []
        values = []
        
        for i, valid_acts in enumerate(valid_actions):
            if not valid_acts:
                actions.append(None)
                log_probs.append(None)
                values.append(None)
                continue
            
            valid_mask = torch.zeros(81, dtype=torch.bool, device=device)
            valid_mask[valid_acts] = True
            
            policy_i = policy[i]  # Get policy for this environment
            policy_i[~valid_mask] = -float('inf')
            probs = F.softmax(policy_i, dim=0)
            
            dist = Categorical(probs=probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            actions.append(action.item())
            log_probs.append(log_prob.item())
            values.append(value[i].item())
        
        return actions, log_probs, values

    def learn(self, last_values, last_dones):
        advantages = torch.zeros_like(self.memory.rewards, device=self.device)
        last_gae_lam = 0
        num_steps_actual = self.memory.ptr
        values_for_gae = self.memory.values[:num_steps_actual]
        rewards_for_gae = self.memory.rewards[:num_steps_actual]
        dones_for_gae = self.memory.dones[:num_steps_actual]

        for t in reversed(range(num_steps_actual)):
            if t == num_steps_actual - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - dones_for_gae[t + 1].float()
                next_values = values_for_gae[t + 1]
            delta = rewards_for_gae[t] + self.gamma * next_values * next_non_terminal - values_for_gae[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        
        returns = advantages[:num_steps_actual] + values_for_gae
        advantages_normalized = (advantages[:num_steps_actual] - advantages[:num_steps_actual].mean()) / (advantages[:num_steps_actual].std() + 1e-8)

        total_value_loss = 0
        total_policy_loss = 0
        total_entropy_loss = 0
        num_batches_processed = 0

        self.policy.train()
        for epoch in range(self.n_epochs):
            batch_generator = self.memory.generate_batches(advantages_normalized, returns)
            
            for batch_states, batch_actions, batch_log_probs, batch_old_values, batch_advantages, batch_returns in batch_generator:
                num_batches_processed += 1
                with autocast(device_type=device_type, enabled=use_amp):
                    new_logits, new_values = self.policy(batch_states)
                    new_values = new_values.squeeze(-1)
                    
                    probs = F.softmax(new_logits, dim=-1)
                    dist = Categorical(probs=probs)
                    
                    new_log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy()
                    
                    logratio = new_log_probs - batch_log_probs
                    ratio = torch.exp(logratio)
                    pg_loss1 = batch_advantages * ratio
                    pg_loss2 = batch_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = -torch.min(pg_loss1, pg_loss2).mean()
                    
                    if self.clip_vloss:
                        value_loss_unclipped = F.mse_loss(new_values, batch_returns, reduction='none')
                        values_clipped = batch_old_values + torch.clamp(new_values - batch_old_values, -self.clip_coef, self.clip_coef)
                        value_loss_clipped = F.mse_loss(values_clipped, batch_returns, reduction='none')
                        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                    else:
                        value_loss = 0.5 * F.mse_loss(new_values, batch_returns)
                    
                    entropy_loss = -entropy.mean()
                    loss = pg_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                self.optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                scaler.step(self.optimizer)
                scaler.update()
                
                total_policy_loss += pg_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
        
        self.memory.clear_memory()
        self.scheduler.step()
        
        avg_value_loss = total_value_loss / num_batches_processed if num_batches_processed > 0 else 0
        avg_policy_loss = total_policy_loss / num_batches_processed if num_batches_processed > 0 else 0
        avg_entropy_loss = total_entropy_loss / num_batches_processed if num_batches_processed > 0 else 0
        
        return avg_value_loss, avg_policy_loss, avg_entropy_loss

# --- Training Loop ---
def train_ppo(total_timesteps=2_000_000, n_steps=1024, batch_size=256, lr=3e-4,
               n_epochs=4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2,
               ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, clip_vloss=True,
               print_interval=20, wandb_log=True, num_envs=4):
    
    try:
        state_shape = (5, BOARD_SIZE, BOARD_SIZE)
        n_actions = BOARD_SIZE * BOARD_SIZE
        total_updates = math.ceil(total_timesteps / (n_steps * num_envs))
        
        agent = PPOAgent(state_shape=state_shape, n_actions=n_actions, lr=lr,
                         n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
                         gamma=gamma, gae_lambda=gae_lambda, clip_coef=clip_coef,
                         ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                         total_updates=total_updates, clip_vloss=clip_vloss, num_envs=num_envs)

        run_name = f"ppo_uttt_{int(time.time())}"
        if wandb_log:
            try:
                wandb.init(project="ultimate-tic-tac-toe-ppo-enhanced", name=run_name, config={
                    "total_timesteps": total_timesteps,
                    "n_steps": n_steps,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "gamma": gamma,
                    "gae_lambda": gae_lambda,
                    "clip_coef": clip_coef,
                    "ent_coef": ent_coef,
                    "vf_coef": vf_coef,
                    "max_grad_norm": max_grad_norm,
                    "n_epochs": n_epochs,
                    "clip_vloss": clip_vloss,
                    "network": "ResNetCNN",
                    "optimizer": "Adam",
                    "device": str(device),
                    "total_updates": total_updates,
                    "num_envs": num_envs
                })
            except Exception as e:
                print(f"Wandb initialization failed: {e}")
                wandb_log = False

        game_states = agent.envs.reset()
        num_timesteps = 0
        num_episodes = 0
        num_updates_done = 0
        recent_episode_rewards = deque(maxlen=100)
        recent_episode_lengths = deque(maxlen=100)
        recent_outcomes = deque(maxlen=100)

        start_time = time.time()
        pbar = tqdm(total=total_timesteps, desc="Training Progress", unit="step")

        while num_timesteps < total_timesteps:
            try:
                agent.policy.train()
                for step in range(n_steps):
                    pbar.update(num_envs)
                    num_timesteps += num_envs
                    
                    actions, log_probs, values = agent.choose_action(game_states)
                    next_game_states = agent.envs.step(actions)
                    
                    rewards = []
                    dones = []
                    for i, (game_state, next_state) in enumerate(zip(game_states, next_game_states)):
                        if actions[i] is None:
                            rewards.append(0.0)
                            dones.append(True)
                            continue
                        
                        reward = 0.0
                        done = next_state["game_over"]
                        if done:
                            winner = next_state["winner"]
                            if winner == PLAYER_O: reward = 1.0
                            elif winner == PLAYER_X: reward = -1.0
                            recent_episode_rewards.append(reward)
                            recent_episode_lengths.append(next_state["move_count"])
                            num_episodes += 1
                        
                        rewards.append(reward)
                        dones.append(done)
                        
                        if log_probs[i] is not None:
                            agent.memory.store_memory(
                                agent.get_state_tensor([game_state]),
                                torch.tensor(actions[i], device=device),
                                torch.tensor(log_probs[i], device=device),
                                torch.tensor(values[i], device=device),
                                torch.tensor(reward, device=device),
                                torch.tensor(done, dtype=torch.bool, device=device)
                            )
                    
                    game_states = next_game_states
                    
                    if num_timesteps >= total_timesteps: break
                
                if agent.memory.ptr == 0: continue
                
                with torch.no_grad():
                    last_values = [agent.get_value(game_state) for game_state in game_states]
                    last_dones = [game_state["game_over"] for game_state in game_states]
                    last_values = torch.tensor(last_values, device=device)
                    last_dones = torch.tensor(last_dones, dtype=torch.float32, device=device)
                
                avg_value_loss, avg_policy_loss, avg_entropy_loss = agent.learn(last_values, last_dones)
                num_updates_done += 1
                
                if num_updates_done % print_interval == 0 and len(recent_episode_rewards) > 0:
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    sps = int(num_timesteps / elapsed_time) if elapsed_time > 0 else 0
                    avg_reward = np.mean(recent_episode_rewards)
                    avg_length = np.mean(recent_episode_lengths)
                    
                    print(f"\n--- Update {num_updates_done} | Timesteps {num_timesteps}/{total_timesteps} ({sps} SPS) ---")
                    print(f"  Avg Reward (Last 100): {avg_reward:.3f}")
                    print(f"  Avg Ep Length (Last 100): {avg_length:.1f}")
                    print(f"  Losses (V/P/Ent): {avg_value_loss:.4f} / {avg_policy_loss:.4f} / {avg_entropy_loss:.4f}")
                    print(f"  Learning Rate: {agent.optimizer.param_groups[0]['lr']:.2e}")
                    
                    if len(recent_episode_rewards) == 100 and avg_reward > agent.best_avg_reward:
                        agent.best_avg_reward = avg_reward
                        save_path = 'ultimate_tic_tac_toe_ppo_best.pth'
                        torch.save(agent.policy.state_dict(), save_path)
                        print(f"  >>> New best model saved with avg reward: {agent.best_avg_reward:.3f} <<<")
                        if wandb_log and wandb.run:
                            wandb.save(save_path)
                    
                    if wandb_log and wandb.run:
                        wandb.log({
                            "rollout/avg_episode_reward_100": avg_reward,
                            "rollout/avg_episode_length_100": avg_length,
                            "train/value_loss": avg_value_loss,
                            "train/policy_loss": avg_policy_loss,
                            "train/entropy_loss": avg_entropy_loss,
                            "train/learning_rate": agent.optimizer.param_groups[0]['lr'],
                            "global_step": num_timesteps,
                            "num_episodes": num_episodes,
                            "num_updates": num_updates_done,
                            "performance/sps": sps
                        })
                    
                    pbar.set_postfix({
                        'Upd': num_updates_done,
                        'AvgRew': f'{avg_reward:.2f}',
                        'VLoss': f'{avg_value_loss:.3f}',
                        'PLoss': f'{avg_policy_loss:.3f}',
                        'Ent': f'{avg_entropy_loss:.3f}',
                        'SPS': sps
                    })
            except Exception as e:
                print(f"Error in training loop: {e}")
                continue

        pbar.close()
        total_training_time = time.time() - start_time
        print(f"\nTraining completed in {total_training_time:.2f} seconds.")

        final_save_path = 'ultimate_tic_tac_toe_ppo_final.pth'
        torch.save(agent.policy.state_dict(), final_save_path)
        print(f"Final model saved to {final_save_path}")
        if wandb_log and wandb.run:
            wandb.save(final_save_path)
            wandb.finish()
    except Exception as e:
        print(f"Error in train_ppo: {e}")
        raise

# --- Main Execution ---
if __name__ == "__main__":
    TOTAL_TIMESTEPS = 1_000_000
    N_STEPS = 1024
    BATCH_SIZE = 256
    LEARNING_RATE = 3e-4
    N_EPOCHS = 4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_COEF = 0.2
    ENT_COEF = 0.01
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    CLIP_VLOSS = True
    PRINT_INTERVAL = 20
    ENABLE_WANDB = True
    NUM_ENVS = 4

    estimated_sps = 2000  # Increased estimate due to optimizations
    estimated_total_seconds = TOTAL_TIMESTEPS / estimated_sps
    estimated_total_hours = estimated_total_seconds / 3600

    print("-" * 40)
    print("Kaggle Training Time Estimate (VERY ROUGH):")
    print(f"  Total Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Estimated Steps/Second (SPS): {estimated_sps:,} (Highly variable!)")
    print(f"  Estimated Training Hours: {estimated_total_hours:.2f} hours")
    print("  Note: Kaggle GPU sessions typically last 9-12 hours.")
    print("        Adjust TOTAL_TIMESTEPS based on available time.")
    print("-" * 40)

    print("Starting training...")
    train_ppo(
        total_timesteps=TOTAL_TIMESTEPS,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_coef=CLIP_COEF,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        clip_vloss=CLIP_VLOSS,
        print_interval=PRINT_INTERVAL,
        wandb_log=ENABLE_WANDB,
        num_envs=NUM_ENVS
    )

    print("\n" + "="*30)
    play_now = input("Training finished. Would you like to play against the *best* trained model? (y/n): ")
    model_to_play = None
    if play_now.lower() == 'y':
        model_to_play = 'ultimate_tic_tac_toe_ppo_best.pth'
    else:
        play_final = input("Play against the *final* trained model? (y/n): ")
        if play_final.lower() == 'y':
            model_to_play = 'ultimate_tic_tac_toe_ppo_final.pth'

    if model_to_play and os.path.exists(model_to_play):
        play_against_model(model_path=model_to_play)
    elif model_to_play:
        print(f"Model file '{model_to_play}' not found. Cannot start game.")
    print("Exiting.")