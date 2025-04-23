import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
from collections import deque, namedtuple
import time
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define constants
BOARD_SIZE = 9
SUB_BOARD_SIZE = 3
EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1

# Training parameters
BATCH_SIZE = 256  # Increased from 128
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05  # Lowered from 0.1
EPSILON_DECAY = 0.9995  # Slower decay
LEARNING_RATE = 0.0005  # Increased from 0.0001
TARGET_UPDATE = 20  # More frequent updates
REPLAY_BUFFER_SIZE = 100000  # Increased from 10000
MCTS_SIMULATIONS = 100  # Increased from 50

# Reward shaping parameters
REWARD_WIN = 10.0
REWARD_LOSE = -10.0
REWARD_DRAW = 0.0
REWARD_SUB_WIN = 1.0
REWARD_SUB_LOSE = -1.0
REWARD_CENTER_CONTROL = 0.2  # Reward for controlling center sub-board
REWARD_MULTIPLE_OPPORTUNITIES = 0.1  # Reward for creating multiple winning opportunities

class UltimateTicTacToeEnv:
    def __init__(self):
        # Initialize the board as a 9x9 grid (0 = empty, 1 = X, -1 = O)
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        
        # Initialize the meta-board (tracking wins in sub-boards)
        self.meta_board = np.zeros((SUB_BOARD_SIZE, SUB_BOARD_SIZE), dtype=np.int8)
        
        # Current player (1 for X, -1 for O)
        self.current_player = PLAYER_X
        
        # Active sub-board for the next move (None means any sub-board)
        self.active_sub_board = None
        
        # Game state
        self.done = False
        self.winner = None
        
        # History for visualization
        self.move_history = []
    
    def reset(self):
        """Reset the game to initial state"""
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.meta_board = np.zeros((SUB_BOARD_SIZE, SUB_BOARD_SIZE), dtype=np.int8)
        self.current_player = PLAYER_X
        self.active_sub_board = None
        self.done = False
        self.winner = None
        self.move_history = []
        return self.get_state()
    
    def get_state(self):
        """Return the current state representation for the agent"""
        # State includes: board state, meta-board state, active sub-board, current player
        return {
            'board': self.board.copy(),
            'meta_board': self.meta_board.copy(),
            'active_sub_board': self.active_sub_board,
            'current_player': self.current_player
        }
    
    def get_valid_moves(self):
        """Return a list of valid move indices (flattened from 0-80)"""
        valid_moves = []
        
        if self.done:
            return valid_moves
            
        # If an active sub-board is specified and not full/won
        if self.active_sub_board is not None:
            sub_row, sub_col = self.active_sub_board
            
            # Check if the sub-board is already won
            if self.meta_board[sub_row, sub_col] != EMPTY:
                # Sub-board is won, can play anywhere that's empty
                return self.get_all_valid_moves()
            
            # Check if the sub-board is full
            start_row, start_col = sub_row * SUB_BOARD_SIZE, sub_col * SUB_BOARD_SIZE
            sub_board_full = True
            
            for i in range(SUB_BOARD_SIZE):
                for j in range(SUB_BOARD_SIZE):
                    row, col = start_row + i, start_col + j
                    if self.board[row, col] == EMPTY:
                        sub_board_full = False
                        valid_moves.append(row * BOARD_SIZE + col)
            
            # If sub-board is full, can play anywhere
            if sub_board_full:
                return self.get_all_valid_moves()
            
            return valid_moves
        else:
            # First move or playing anywhere - get all valid moves
            return self.get_all_valid_moves()
    
    def get_all_valid_moves(self):
        """Return all valid moves across the entire board"""
        valid_moves = []
        
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                # Check if cell is empty
                if self.board[i, j] == EMPTY:
                    # Check if the sub-board is not already won
                    sub_row, sub_col = i // SUB_BOARD_SIZE, j // SUB_BOARD_SIZE
                    if self.meta_board[sub_row, sub_col] == EMPTY:
                        valid_moves.append(i * BOARD_SIZE + j)
        
        return valid_moves
    
    def step(self, action):
        """Make a move and return the next state, reward, and done flag"""
        if self.done:
            return self.get_state(), 0, True, {"winner": self.winner}
        
        # Convert action to row and column
        row = action // BOARD_SIZE
        col = action % BOARD_SIZE
        
        # Check if move is valid
        sub_row, sub_col = row // SUB_BOARD_SIZE, col // SUB_BOARD_SIZE
        
        # If active sub-board is specified, check if the move is in the correct sub-board
        if self.active_sub_board is not None:
            act_sub_row, act_sub_col = self.active_sub_board
            if (sub_row != act_sub_row or sub_col != act_sub_col) and self.meta_board[act_sub_row, act_sub_col] == EMPTY:
                # Sub-board not full and not playing in the right sub-board
                # This is an invalid move, return current state with negative reward
                return self.get_state(), -10, False, {"invalid_move": True}
        
        # Check if cell is already occupied
        if self.board[row, col] != EMPTY:
            return self.get_state(), -10, False, {"invalid_move": True}
        
        # Check if the sub-board is already won
        if self.meta_board[sub_row, sub_col] != EMPTY:
            return self.get_state(), -10, False, {"invalid_move": True}
        
        # Make the move
        self.board[row, col] = self.current_player
        self.move_history.append((row, col, self.current_player))
        
        # Determine the next active sub-board
        next_sub_row, next_sub_col = row % SUB_BOARD_SIZE, col % SUB_BOARD_SIZE
        
        # Check if the sub-board is won after this move
        start_row, start_col = sub_row * SUB_BOARD_SIZE, sub_col * SUB_BOARD_SIZE
        sub_board = self.board[start_row:start_row+SUB_BOARD_SIZE, start_col:start_col+SUB_BOARD_SIZE]
        sub_board_winner = self.check_winner(sub_board)
        
        reward = 0
        
        if sub_board_winner != EMPTY:
            # Sub-board is won
            self.meta_board[sub_row, sub_col] = sub_board_winner
            reward = 1 if sub_board_winner == self.current_player else -1
            
            # Check if the game is won
            meta_winner = self.check_winner(self.meta_board)
            if meta_winner != EMPTY:
                self.done = True
                self.winner = meta_winner
                reward = 10 if meta_winner == self.current_player else -10
        
        # Check if the next sub-board is already won or full
        if self.meta_board[next_sub_row, next_sub_col] != EMPTY or \
           self.is_sub_board_full(next_sub_row, next_sub_col):
            self.active_sub_board = None  # Can play anywhere
        else:
            self.active_sub_board = (next_sub_row, next_sub_col)
        
        # Check if the game is a draw
        if not self.done and self.is_board_full():
            self.done = True
            self.winner = EMPTY  # Draw
        
        # Switch players
        self.current_player *= -1
        
        return self.get_state(), reward, self.done, {"winner": self.winner}
    
    def check_winner(self, board):
        """Check if there's a winner in the given board"""
        # Check rows
        for i in range(SUB_BOARD_SIZE):
            if abs(np.sum(board[i, :])) == SUB_BOARD_SIZE:
                return np.sign(np.sum(board[i, :]))
        
        # Check columns
        for i in range(SUB_BOARD_SIZE):
            if abs(np.sum(board[:, i])) == SUB_BOARD_SIZE:
                return np.sign(np.sum(board[:, i]))
        
        # Check diagonals
        if abs(np.trace(board)) == SUB_BOARD_SIZE:
            return np.sign(np.trace(board))
        
        if abs(np.trace(np.fliplr(board))) == SUB_BOARD_SIZE:
            return np.sign(np.trace(np.fliplr(board)))
        
        return EMPTY
    
    def is_sub_board_full(self, sub_row, sub_col):
        """Check if a sub-board is full"""
        start_row, start_col = sub_row * SUB_BOARD_SIZE, sub_col * SUB_BOARD_SIZE
        sub_board = self.board[start_row:start_row+SUB_BOARD_SIZE, start_col:start_col+SUB_BOARD_SIZE]
        return not (EMPTY in sub_board)
    
    def is_board_full(self):
        """Check if the entire board is full"""
        for sub_row in range(SUB_BOARD_SIZE):
            for sub_col in range(SUB_BOARD_SIZE):
                if self.meta_board[sub_row, sub_col] == EMPTY and not self.is_sub_board_full(sub_row, sub_col):
                    return False
        return True
    
    def render(self):
        """Render the board to the console"""
        print("\n" + "-" * 25)
        for i in range(BOARD_SIZE):
            row_str = "| "
            for j in range(BOARD_SIZE):
                # Add vertical divider every 3 columns
                if j > 0 and j % SUB_BOARD_SIZE == 0:
                    row_str += "| "
                
                # Add X, O, or space
                if self.board[i, j] == PLAYER_X:
                    row_str += "X "
                elif self.board[i, j] == PLAYER_O:
                    row_str += "O "
                else:
                    row_str += ". "
            row_str += "|"
            print(row_str)
            
            # Add horizontal divider every 3 rows
            if (i + 1) % SUB_BOARD_SIZE == 0 and i < BOARD_SIZE - 1:
                print("-" * 25)
        print("-" * 25)
        
        # Display meta-board status
        print("\nMeta-board:")
        for i in range(SUB_BOARD_SIZE):
            row_str = "| "
            for j in range(SUB_BOARD_SIZE):
                if self.meta_board[i, j] == PLAYER_X:
                    row_str += "X "
                elif self.meta_board[i, j] == PLAYER_O:
                    row_str += "O "
                else:
                    row_str += ". "
            row_str += "|"
            print(row_str)
        
        # Display game status
        if self.done:
            if self.winner == PLAYER_X:
                print("\nPlayer X wins!")
            elif self.winner == PLAYER_O:
                print("\nPlayer O wins!")
            else:
                print("\nGame ended in a draw!")
        else:
            print(f"\nCurrent player: {'X' if self.current_player == PLAYER_X else 'O'}")
            if self.active_sub_board is not None:
                print(f"Active sub-board: {self.active_sub_board}")
            else:
                print("Active sub-board: Any")

# Improved DQN model with attention and residual connections
class ImprovedDQN(nn.Module):
    def __init__(self):
        super(ImprovedDQN, self).__init__()
        
        # Feature extraction
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        
        # Attention mechanism
        self.attention = AttentionBlock(64)
        
        # Policy head
        self.policy_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.policy_fc = nn.Linear(32 * 9 * 9, 81)
        
        # Value head
        self.value_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.value_fc1 = nn.Linear(32 * 9 * 9, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        # Attention
        x = self.attention(x)
        
        # Policy head
        policy = self.policy_conv(x)
        policy = policy.view(-1, 32 * 9 * 9)
        policy = self.policy_fc(policy)
        
        # Value head
        value = self.value_conv(x)
        value = value.view(-1, 32 * 9 * 9)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Calculate attention scores
        query = self.query(x).view(batch_size, -1, height * width)
        key = self.key(x).view(batch_size, -1, height * width)
        attention = torch.bmm(query.transpose(1, 2), key)
        attention = F.softmax(attention, dim=2)
        
        # Apply attention
        value = self.value(x).view(batch_size, -1, height * width)
        out = torch.bmm(value, attention.transpose(1, 2))
        out = out.view(batch_size, channels, height, width)
        
        return self.gamma * out + x

# Prioritized Experience Replay
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.memory = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.memory) == 0:
            return None
        
        priorities = self.priorities[:len(self.memory)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
        self.max_priority = max(self.max_priority, max(priorities))

# Improved DQN Agent with MCTS
class ImprovedDQNAgent:
    def __init__(self, state_dim, action_dim, replay_buffer_size=REPLAY_BUFFER_SIZE, batch_size=BATCH_SIZE,
                 gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY,
                 learning_rate=LEARNING_RATE, target_update=TARGET_UPDATE, mcts_simulations=MCTS_SIMULATIONS):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.mcts_simulations = mcts_simulations
        self.steps = 0
        
        # Initialize policy and target networks
        self.policy_net = ImprovedDQN().to(device)
        self.target_net = ImprovedDQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = PrioritizedReplayBuffer(replay_buffer_size)
        
        # Training metrics
        self.losses = []
        self.episode_rewards = []
        self.win_rate = []
        self.current_episode_reward = 0
        
        # Add new attributes for opponent learning
        self.opponent_patterns = {}  # Store opponent move patterns
        self.opponent_memory = deque(maxlen=1000)  # Store recent opponent moves
        self.learning_rate_opponent = 0.001  # Higher learning rate for opponent patterns
        self.pattern_window = 3  # Number of moves to consider for patterns
        
        # Add new attributes for dynamic learning
        self.base_learning_rate = learning_rate
        self.min_learning_rate = learning_rate * 0.1
        self.learning_rate_decay = 0.9995
        self.win_rate_threshold = 0.5
        self.temperature = 1.0
        self.temperature_decay = 0.999
        self.min_temperature = 0.1
        
        # Advanced reward shaping parameters
        self.reward_weights = {
            'win': 10.0,
            'lose': -10.0,
            'draw': 0.0,
            'sub_win': 1.0,
            'sub_lose': -1.0,
            'center_control': 0.5,
            'multiple_opportunities': 0.3,
            'block_opponent': 0.4,
            'create_fork': 0.6,
            'prevent_fork': 0.5
        }
    
    @property
    def learning_rate(self):
        """Get the current learning rate from the optimizer"""
        return self.optimizer.param_groups[0]['lr']
    
    @learning_rate.setter
    def learning_rate(self, value):
        """Set the learning rate in the optimizer"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = value
    
    def preprocess_state(self, state):
        """Convert the game state to a tensor representation"""
        board = state['board']
        meta_board = state['meta_board']
        active_sub_board = state['active_sub_board']
        
        # Create channels for X, O, meta board, and active sub-board
        x_channel = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        o_channel = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        meta_channel = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        active_channel = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        
        # Fill X and O channels
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i, j] == PLAYER_X:
                    x_channel[i, j] = 1
                elif board[i, j] == PLAYER_O:
                    o_channel[i, j] = 1
        
        # Fill meta board channel
        for i in range(SUB_BOARD_SIZE):
            for j in range(SUB_BOARD_SIZE):
                if meta_board[i, j] != EMPTY:
                    value = 1 if meta_board[i, j] == PLAYER_X else -1
                    start_row, start_col = i * SUB_BOARD_SIZE, j * SUB_BOARD_SIZE
                    meta_channel[start_row:start_row+SUB_BOARD_SIZE, start_col:start_col+SUB_BOARD_SIZE] = value
        
        # Fill active sub-board channel
        if active_sub_board is not None:
            sub_row, sub_col = active_sub_board
            start_row, start_col = sub_row * SUB_BOARD_SIZE, sub_col * SUB_BOARD_SIZE
            active_channel[start_row:start_row+SUB_BOARD_SIZE, start_col:start_col+SUB_BOARD_SIZE] = 1
        
        # Stack channels
        state_tensor = np.stack([x_channel, o_channel, meta_channel, active_channel], axis=0)
        return torch.FloatTensor(state_tensor).unsqueeze(0).to(device)
    
    def select_action(self, state, valid_moves, training=True):
        """Select action with temperature-based exploration"""
        if not valid_moves:
            return None
        
        if training and random.random() < self.epsilon:
            # Temperature-based exploration
            q_values = self.get_q_values(state)
            valid_q_values = [q_values[move] for move in valid_moves]
            
            # Apply temperature to Q-values
            exp_q_values = np.exp(np.array(valid_q_values) / self.temperature)
            probs = exp_q_values / np.sum(exp_q_values)
            
            # Sample action based on temperature-scaled probabilities
            return valid_moves[np.random.choice(len(valid_moves), p=probs)]
        else:
            # Exploitation
            q_values = self.get_q_values(state)
            valid_q_values = [q_values[move] for move in valid_moves]
            return valid_moves[np.argmax(valid_q_values)]
    
    def get_q_values(self, state):
        """Get Q-values for all actions"""
        state_tensor = self.preprocess_state(state)
        with torch.no_grad():
            policy, value = self.policy_net(state_tensor)
            # Use policy head for action selection
            q_values = policy.squeeze(0)  # Remove batch dimension
        return q_values.cpu().numpy()
    
    def optimize_model(self):
        """Perform a single step of optimization on the policy network"""
        if len(self.memory.memory) < self.batch_size:
            return 0.0
        
        # Sample a batch of transitions from memory
        samples, indices, weights = self.memory.sample(self.batch_size)
        if samples is None:
            return 0.0
        
        batch = list(zip(*samples))
        state_batch = torch.cat(batch[0])
        action_batch = torch.tensor(batch[1], dtype=torch.long).unsqueeze(1).to(device)
        reward_batch = torch.tensor(batch[2], dtype=torch.float).to(device)
        next_state_batch = torch.cat(batch[3])
        done_batch = torch.tensor(batch[4], dtype=torch.float).to(device)
        weights = torch.tensor(weights, dtype=torch.float).to(device)
        
        # Compute Q(s_t, a) for all actions
        policy, value = self.policy_net(state_batch)
        state_action_values = policy.gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states
        with torch.no_grad():
            next_policy, next_value = self.target_net(next_state_batch)
            next_state_values = next_value.squeeze()
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma * (1 - done_batch)) + reward_batch
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        loss = (loss * weights).mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Only clamp gradients if they exist
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()
        
        # Update priorities
        priorities = (state_action_values - expected_state_action_values.unsqueeze(1)).abs().detach().cpu().numpy()
        self.memory.update_priorities(indices, priorities)
        
        self.losses.append(loss.item())
        return loss.item()
    
    def decay_epsilon(self):
        """Decay the exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """Update the target network parameters"""
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, path):
        """Save the policy network parameters"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load the policy network parameters"""
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['policy_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.steps = checkpoint['steps']
            print(f"Model loaded from {path}")
            return True
        return False

    def count_winning_opportunities(self, state, player=None):
        """Count the number of potential winning moves for a given player"""
        if player is None:
            player = state['current_player']
        
        opportunities = 0
        
        # Check each sub-board
        for i in range(SUB_BOARD_SIZE):
            for j in range(SUB_BOARD_SIZE):
                if state['meta_board'][i, j] == EMPTY:
                    sub_board = state['board'][i*SUB_BOARD_SIZE:(i+1)*SUB_BOARD_SIZE, 
                                             j*SUB_BOARD_SIZE:(j+1)*SUB_BOARD_SIZE]
                    opportunities += self.count_sub_board_opportunities(sub_board, player)
        
        return opportunities

    def count_sub_board_opportunities(self, sub_board, player):
        """Count winning opportunities in a sub-board for a given player"""
        opportunities = 0
        
        # Check rows
        for i in range(SUB_BOARD_SIZE):
            if np.sum(sub_board[i, :] == player) == 2 and np.sum(sub_board[i, :] == EMPTY) == 1:
                opportunities += 1
        
        # Check columns
        for i in range(SUB_BOARD_SIZE):
            if np.sum(sub_board[:, i] == player) == 2 and np.sum(sub_board[:, i] == EMPTY) == 1:
                opportunities += 1
        
        # Check diagonals
        if np.sum(np.diag(sub_board) == player) == 2 and np.sum(np.diag(sub_board) == EMPTY) == 1:
            opportunities += 1
        if np.sum(np.diag(np.fliplr(sub_board)) == player) == 2 and np.sum(np.diag(np.fliplr(sub_board)) == EMPTY) == 1:
            opportunities += 1
        
        return opportunities

    def calculate_advanced_reward(self, state, next_state, reward, done):
        """Calculate shaped reward with advanced strategic considerations"""
        shaped_reward = reward
        
        # Center control reward
        if state['meta_board'][1, 1] == EMPTY and next_state['meta_board'][1, 1] != EMPTY:
            if next_state['meta_board'][1, 1] == state['current_player']:
                shaped_reward += self.reward_weights['center_control']
        
        # Multiple opportunities reward
        current_opportunities = self.count_winning_opportunities(state)
        next_opportunities = self.count_winning_opportunities(next_state)
        if next_opportunities > current_opportunities:
            shaped_reward += self.reward_weights['multiple_opportunities'] * (next_opportunities - current_opportunities)
        
        # Block opponent reward
        opponent_opportunities_before = self.count_winning_opportunities(state, -state['current_player'])
        opponent_opportunities_after = self.count_winning_opportunities(next_state, -state['current_player'])
        if opponent_opportunities_after < opponent_opportunities_before:
            shaped_reward += self.reward_weights['block_opponent'] * (opponent_opportunities_before - opponent_opportunities_after)
        
        # Fork creation reward
        if self.detect_fork_creation(state, next_state):
            shaped_reward += self.reward_weights['create_fork']
        
        # Fork prevention reward
        if self.detect_fork_prevention(state, next_state):
            shaped_reward += self.reward_weights['prevent_fork']
        
        return shaped_reward
    
    def detect_fork_creation(self, state, next_state):
        """Detect if a fork (multiple winning opportunities) was created"""
        current_forks = self.count_forks(state)
        next_forks = self.count_forks(next_state)
        return next_forks > current_forks
    
    def detect_fork_prevention(self, state, next_state):
        """Detect if an opponent's fork was prevented"""
        current_opponent_forks = self.count_forks(state, -state['current_player'])
        next_opponent_forks = self.count_forks(next_state, -state['current_player'])
        return next_opponent_forks < current_opponent_forks
    
    def count_forks(self, state, player=None):
        """Count the number of forks (multiple winning opportunities)"""
        if player is None:
            player = state['current_player']
        
        forks = 0
        for i in range(SUB_BOARD_SIZE):
            for j in range(SUB_BOARD_SIZE):
                if state['meta_board'][i, j] == EMPTY:
                    sub_board = state['board'][i*SUB_BOARD_SIZE:(i+1)*SUB_BOARD_SIZE, 
                                             j*SUB_BOARD_SIZE:(j+1)*SUB_BOARD_SIZE]
                    winning_lines = self.count_winning_lines(sub_board, player)
                    if winning_lines >= 2:  # Fork exists if there are 2 or more winning lines
                        forks += 1
        return forks
    
    def count_winning_lines(self, sub_board, player):
        """Count the number of potential winning lines in a sub-board"""
        lines = 0
        
        # Check rows
        for i in range(SUB_BOARD_SIZE):
            if np.sum(sub_board[i, :] == player) == 2 and np.sum(sub_board[i, :] == EMPTY) == 1:
                lines += 1
        
        # Check columns
        for i in range(SUB_BOARD_SIZE):
            if np.sum(sub_board[:, i] == player) == 2 and np.sum(sub_board[:, i] == EMPTY) == 1:
                lines += 1
        
        # Check diagonals
        if np.sum(np.diag(sub_board) == player) == 2 and np.sum(np.diag(sub_board) == EMPTY) == 1:
            lines += 1
        if np.sum(np.diag(np.fliplr(sub_board)) == player) == 2 and np.sum(np.diag(np.fliplr(sub_board)) == EMPTY) == 1:
            lines += 1
        
        return lines
    
    def learn_from_opponent(self, opponent_moves, game_result):
        """Learn from opponent's moves and game outcome"""
        # Store opponent moves
        self.opponent_memory.extend(opponent_moves)
        
        # Update opponent patterns
        for i in range(len(opponent_moves) - self.pattern_window + 1):
            pattern = tuple(opponent_moves[i:i + self.pattern_window])
            if pattern not in self.opponent_patterns:
                self.opponent_patterns[pattern] = {'count': 0, 'success': 0}
            self.opponent_patterns[pattern]['count'] += 1
            if game_result == 'win':
                self.opponent_patterns[pattern]['success'] += 1
    
    def predict_opponent_move(self, state):
        """Predict opponent's next move based on learned patterns"""
        if not self.opponent_memory:
            return None
            
        # Get recent moves
        recent_moves = list(self.opponent_memory)[-self.pattern_window+1:]
        
        # Find matching patterns
        matching_patterns = []
        for pattern, stats in self.opponent_patterns.items():
            if pattern[:-1] == tuple(recent_moves):
                matching_patterns.append((pattern[-1], stats['success'] / stats['count']))
        
        if matching_patterns:
            # Return most successful pattern's next move
            return max(matching_patterns, key=lambda x: x[1])[0]
        return None

    def adjust_learning_rate(self, win_rate):
        """Dynamically adjust learning rate based on performance"""
        if win_rate < self.win_rate_threshold:
            # Increase learning rate when performing poorly
            new_lr = min(self.learning_rate * 1.5, self.base_learning_rate * 2)
        else:
            # Decrease learning rate when performing well
            new_lr = max(self.learning_rate * self.learning_rate_decay, self.min_learning_rate)
        
        self.learning_rate = new_lr
    
    def adjust_temperature(self):
        """Adjust exploration temperature"""
        self.temperature = max(self.temperature * self.temperature_decay, self.min_temperature)

# Training function with curriculum learning
def train_agent(agent, env, num_episodes=10000, early_stopping_window=100, early_stopping_threshold=0.95):
    """Train the agent with advanced learning strategies"""
    start_time = time.time()
    reward_window = deque(maxlen=early_stopping_window)
    win_window = deque(maxlen=early_stopping_window)
    best_win_rate = 0.0
    no_improvement_count = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            valid_moves = env.get_valid_moves()
            if not valid_moves:
                break
            
            action = agent.select_action(state, valid_moves, training=True)
            if action is None:
                break
            
            next_state, reward, done, info = env.step(action)
            
            # Calculate advanced reward
            shaped_reward = agent.calculate_advanced_reward(state, next_state, reward, done)
            
            # Store transition
            state_tensor = agent.preprocess_state(state)
            next_state_tensor = agent.preprocess_state(next_state)
            agent.memory.push(state_tensor, action, shaped_reward, next_state_tensor, done)
            
            state = next_state
            episode_reward += shaped_reward
            steps += 1
            
            # Optimize model
            if steps % 4 == 0:
                loss = agent.optimize_model()
            
            # Update target network
            agent.update_target_network()
            
            # Update steps
            agent.steps += 1
        
        # Update metrics
        agent.episode_rewards.append(episode_reward)
        reward_window.append(episode_reward)
        
        if 'winner' in info and info['winner'] is not None:
            won = (info['winner'] == PLAYER_X)
            win_window.append(1 if won else 0)
        else:
            win_window.append(0)
        
        # Calculate win rate
        if len(win_window) == early_stopping_window:
            win_rate = sum(win_window) / len(win_window)
            
            # Adjust learning parameters based on performance
            agent.adjust_learning_rate(win_rate)
            agent.adjust_temperature()
            
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                no_improvement_count = 0
                agent.save_model('best_improved_model.pt')
            else:
                no_improvement_count += 1
        
        # Print progress
        if episode % 50 == 0:
            elapsed_time = time.time() - start_time
            win_rate_str = f"{sum(win_window) / len(win_window):.2f}" if win_window else "N/A"
            avg_reward_str = f"{sum(reward_window) / len(reward_window):.2f}" if reward_window else "N/A"
            
            print(f"Episode {episode}/{num_episodes}, "
                  f"Win Rate: {win_rate_str}, "
                  f"Avg Reward: {avg_reward_str}, "
                  f"Steps: {steps}, "
                  f"LR: {agent.learning_rate:.6f}, "
                  f"Temp: {agent.temperature:.2f}, "
                  f"Time: {elapsed_time:.1f}s")
        
        # Early stopping
        if no_improvement_count >= 500:
            print(f"Stopping due to no improvement for {no_improvement_count} episodes")
            break
    
    return agent

def play_against_agent(agent, env, continue_learning=True):
    """Let the human play against the trained agent"""
    state = env.reset()
    done = False
    human_player = PLAYER_O
    agent_player = PLAYER_X
    opponent_moves = []
    
    print("\nPlaying against the trained agent")
    print("You are playing as O")
    print("Enter moves as row,col (0-8,0-8)")
    
    while not done:
        env.render()
        
        current_player = env.current_player
        
        if current_player == human_player:
            # Human's turn
            valid_moves = env.get_valid_moves()
            if not valid_moves:
                print("No valid moves available. Game over.")
                break
            
            # Get human move
            while True:
                try:
                    move_str = input("Enter your move (row,col): ")
                    row, col = map(int, move_str.split(','))
                    action = row * BOARD_SIZE + col
                    
                    if action in valid_moves:
                        break
                    print("Invalid move. Try again.")
                except ValueError:
                    print("Invalid input. Please use format 'row,col' with values between 0 and 8.")
            
            # Store human move
            if len(opponent_moves) < 3:
                opponent_moves.append(action)
            
            # Make the move
            next_state, reward, done, info = env.step(action)
            
            if continue_learning:
                # Learn from human move
                state_tensor = agent.preprocess_state(state)
                next_state_tensor = agent.preprocess_state(next_state)
                agent.memory.push(state_tensor, action, reward, next_state_tensor, done)
                agent.learn_from_opponent(opponent_moves, 'loss' if reward < 0 else 'win')
            
            state = next_state
            
        else:
            # Agent's turn
            print("Agent is thinking...")
            
            valid_moves = env.get_valid_moves()
            if not valid_moves:
                print("No valid moves available. Game over.")
                break
            
            # Try to predict human's move
            predicted_move = agent.predict_opponent_move(state)
            if predicted_move is not None and predicted_move in valid_moves:
                action = predicted_move
            else:
                action = agent.select_action(state, valid_moves, training=False)
            
            if action is None:
                print("Agent couldn't find a valid move. Game over.")
                break
            
            # Make the move
            next_state, reward, done, info = env.step(action)
            state = next_state
    
    # Game over
    env.render()
    
    if env.winner == human_player:
        print("Congratulations! You win!")
        if continue_learning:
            agent.learn_from_opponent(opponent_moves, 'loss')
    elif env.winner == agent_player:
        print("Agent wins. Better luck next time!")
    else:
        print("Game ended in a draw.")
    
    if continue_learning:
        agent.save_model('updated_improved_model.pt')
        print("Agent has learned from this game. Model updated.")

def plot_training_metrics(win_rates, episode_rewards):
    """Plot the training metrics"""
    plt.figure(figsize=(15, 5))
    
    # Plot win rate
    plt.subplot(1, 2, 1)
    plt.plot(win_rates)
    plt.title('Win Rate')
    plt.xlabel('Episodes (x100)')
    plt.ylabel('Win Rate')
    
    # Plot episode rewards
    plt.subplot(1, 2, 2)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.tight_layout()
    plt.savefig('improved_training_metrics.png')
    plt.close()

if __name__ == "__main__":
    # Create environment and agent
    env = UltimateTicTacToeEnv()
    agent = ImprovedDQNAgent(
        state_dim=(4, BOARD_SIZE, BOARD_SIZE),
        action_dim=BOARD_SIZE * BOARD_SIZE,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        learning_rate=LEARNING_RATE,
        target_update=TARGET_UPDATE,
        mcts_simulations=MCTS_SIMULATIONS
    )
    
    # Train the agent
    train_agent(agent, env, num_episodes=10000) 