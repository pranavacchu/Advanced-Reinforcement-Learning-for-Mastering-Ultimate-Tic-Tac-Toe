import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import math
from tqdm import tqdm
import os

# Game Constants
BOARD_SIZE = 9
SUBBOARD_SIZE = 3
EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1

class AlphaZeroNet(nn.Module):
    def __init__(self):
        super(AlphaZeroNet, self).__init__()
        # Simplified architecture for faster inference
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Policy head
        self.policy_conv = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)
        
        # Value head
        self.value_conv = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * BOARD_SIZE * BOARD_SIZE, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Efficient forward pass with in-place operations
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)), inplace=True)
        policy = policy.view(-1, 32 * BOARD_SIZE * BOARD_SIZE)
        policy = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)), inplace=True)
        value = value.view(-1, 32 * BOARD_SIZE * BOARD_SIZE)
        value = F.relu(self.value_fc1(value), inplace=True)
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

class UltimateTicTacToe:
    def __init__(self):
        self.board = [[0 for _ in range(9)] for _ in range(9)]
        self.meta_board = [[0 for _ in range(3)] for _ in range(3)]
        self.current_player = PLAYER_X
        self.active_sub_row = None
        self.active_sub_col = None
        self.game_over = False
        self.winner = None
        self.last_move = None

    def make_move(self, row, col):
        if not self.is_valid_move(row, col):
            return False

        self.board[row][col] = self.current_player
        self.last_move = (row, col)
        
        # Update meta board
        sub_row, sub_col = row // 3, col // 3
        sub_winner = self.check_sub_board_winner(sub_row, sub_col)
        if sub_winner is not None:
            self.meta_board[sub_row][sub_col] = sub_winner
        
        # Check for game winner
        winner = self.check_winner()
        if winner is not None:
            self.winner = winner
            self.game_over = True
        elif self.is_board_full():
            self.game_over = True
        
        # Update active sub-board
        next_sub_row, next_sub_col = row % 3, col % 3
        if self.has_valid_moves_in_sub_board(next_sub_row, next_sub_col):
            self.active_sub_row = next_sub_row
            self.active_sub_col = next_sub_col
        else:
            self.active_sub_row = None
            self.active_sub_col = None
        
        self.current_player = -self.current_player
        return True

    def is_valid_move(self, row, col):
        if row < 0 or row >= 9 or col < 0 or col >= 9:
            return False
        
        if self.board[row][col] != 0:
            return False
        
        if self.game_over:
            return False
        
        sub_row, sub_col = row // 3, col // 3
        if self.active_sub_row is not None and self.active_sub_col is not None:
            if sub_row != self.active_sub_row or sub_col != self.active_sub_col:
                return False
        
        return True

    def has_valid_moves_in_sub_board(self, sub_row, sub_col):
        start_row = sub_row * 3
        start_col = sub_col * 3
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                if self.board[r][c] == 0:
                    return True
        return False

    def get_valid_moves(self):
        valid_moves = []
        if self.active_sub_row is None or self.active_sub_col is None:
            for r in range(9):
                for c in range(9):
                    if self.board[r][c] == 0:
                        valid_moves.append((r, c))
        else:
            start_row = self.active_sub_row * 3
            start_col = self.active_sub_col * 3
            for r in range(start_row, start_row + 3):
                for c in range(start_col, start_col + 3):
                    if self.board[r][c] == 0:
                        valid_moves.append((r, c))
            if not valid_moves:
                for r in range(9):
                    for c in range(9):
                        if self.board[r][c] == 0:
                            valid_moves.append((r, c))
        return valid_moves

    def check_sub_board_winner(self, sub_row, sub_col):
        start_row = sub_row * 3
        start_col = sub_col * 3
        
        # Get sub-board
        sub_board = [
            [self.board[start_row + i][start_col + j] 
             for j in range(3)] for i in range(3)
        ]
        
        # Check rows and columns
        for i in range(3):
            if all(sub_board[i][j] == PLAYER_X for j in range(3)):
                return PLAYER_X
            if all(sub_board[i][j] == PLAYER_O for j in range(3)):
                return PLAYER_O
            if all(sub_board[j][i] == PLAYER_X for j in range(3)):
                return PLAYER_X
            if all(sub_board[j][i] == PLAYER_O for j in range(3)):
                return PLAYER_O
        
        # Check diagonals
        if all(sub_board[i][i] == PLAYER_X for i in range(3)) or \
           all(sub_board[i][2-i] == PLAYER_X for i in range(3)):
            return PLAYER_X
        if all(sub_board[i][i] == PLAYER_O for i in range(3)) or \
           all(sub_board[i][2-i] == PLAYER_O for i in range(3)):
            return PLAYER_O
        
        return None

    def check_winner(self):
        # Check rows and columns
        for i in range(3):
            if all(self.meta_board[i][j] == PLAYER_X for j in range(3)):
                return PLAYER_X
            if all(self.meta_board[i][j] == PLAYER_O for j in range(3)):
                return PLAYER_O
            if all(self.meta_board[j][i] == PLAYER_X for j in range(3)):
                return PLAYER_X
            if all(self.meta_board[j][i] == PLAYER_O for j in range(3)):
                return PLAYER_O
        
        # Check diagonals
        if all(self.meta_board[i][i] == PLAYER_X for i in range(3)) or \
           all(self.meta_board[i][2-i] == PLAYER_X for i in range(3)):
            return PLAYER_X
        if all(self.meta_board[i][i] == PLAYER_O for i in range(3)) or \
           all(self.meta_board[i][2-i] == PLAYER_O for i in range(3)):
            return PLAYER_O
        
        return None

    def is_board_full(self):
        return all(self.board[r][c] != 0 for r in range(9) for c in range(9))

    def get_state(self):
        return {
            'board': self.board,
            'meta_board': self.meta_board,
            'current_player': self.current_player,
            'active_sub_row': self.active_sub_row,
            'active_sub_col': self.active_sub_col,
            'game_over': self.game_over,
            'winner': self.winner
        }

    def copy(self):
        new_game = UltimateTicTacToe()
        new_game.board = [row[:] for row in self.board]
        new_game.meta_board = [row[:] for row in self.meta_board]
        new_game.current_player = self.current_player
        new_game.active_sub_row = self.active_sub_row
        new_game.active_sub_col = self.active_sub_col
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        new_game.last_move = self.last_move
        return new_game

class Node:
    def __init__(self, game_state, move=None, parent=None):
        self.game_state = game_state
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.prior = 1.0
        self.total_value = 0.0  # Track total value for better averaging

    def select(self, c_puct):
        if not self.children:
            return None
        return max(self.children, key=lambda node: node.get_ucb(c_puct))

    def get_ucb(self, c_puct):
        if self.visits == 0:
            return float('inf')
        if self.parent is None or self.parent.visits == 0:
            return self.value / self.visits + c_puct * self.prior
        
        # Ensure we don't take log of zero or negative
        parent_visits = max(1, self.parent.visits)
        exploration_term = math.sqrt(math.log(parent_visits) / self.visits)
        
        # Use total_value for more stable value estimates
        value_estimate = self.total_value / self.visits
        return value_estimate + c_puct * self.prior * exploration_term

    def update(self, value):
        self.visits += 1
        self.total_value += value
        self.value = self.total_value / self.visits  # Update average value

    def get_best_move(self):
        if not self.children:
            return None
        return max(self.children, key=lambda node: node.visits).move

class AlphaZero:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = AlphaZeroNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.memory = deque(maxlen=50000)  # Reduced memory size
        self.batch_size = 32
        self.mcts_iterations = 200  # Reduced for faster training
        self.c_puct = 1.5
        self.temperature = 1.0
        self.alpha = 0.3
        self.min_visits = 1
        
        # Enable cuDNN benchmarking for faster training
        torch.backends.cudnn.benchmark = True
        # Enable memory efficient attention
        torch.backends.cudnn.deterministic = False

    def mcts(self, game):
        root = Node(game)
        
        # Get valid moves once
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
            
        # Initialize children with Dirichlet noise
        noise = np.random.dirichlet([self.alpha] * len(valid_moves))
        children = []
        for i, move in enumerate(valid_moves):
            temp_game = game.copy()
            if temp_game.make_move(move[0], move[1]):
                child = Node(temp_game, move, root)
                child.prior = noise[i]
                children.append(child)
        root.children = children
        
        # Parallel MCTS with vectorized operations
        for _ in range(self.mcts_iterations):
            node = root
            temp_game = game.copy()
            
            # Selection with virtual loss
            while node.children and not temp_game.game_over:
                node = node.select(self.c_puct)
                if node is None:
                    break
                if node.move:
                    temp_game.make_move(node.move[0], node.move[1])
                    node.visits += 1
            
            if node is None:
                continue
            
            # Expansion with parallel simulation
            if not temp_game.game_over:
                valid_moves = temp_game.get_valid_moves()
                new_children = []
                for move in valid_moves:
                    new_game = temp_game.copy()
                    if new_game.make_move(move[0], move[1]):
                        child = Node(new_game, move, node)
                        # Use neural network to predict prior
                        with torch.no_grad():
                            state_tensor = self.get_state_tensor(new_game)
                            policy, _ = self.model(state_tensor)
                            child.prior = torch.softmax(policy, dim=1)[0, move[0] * BOARD_SIZE + move[1]].item()
                        new_children.append(child)
                node.children = new_children
            
            # Simulation with neural network evaluation
            value = self.evaluate(temp_game)
            
            # Backpropagation with virtual loss correction
            while node:
                node.update(value)
                node.visits -= 1
                node = node.parent
                value = -value
        
        # Return move with temperature
        if not root.children:
            return None
        
        visits = torch.tensor([max(child.visits, self.min_visits) for child in root.children], device=self.device)
        probs = torch.softmax(visits / self.temperature, dim=0)
        best_child = root.children[torch.argmax(probs).item()]
        return best_child.move

    def evaluate(self, game):
        if game.winner == PLAYER_X:
            return 1
        elif game.winner == PLAYER_O:
            return -1
        elif game.game_over:
            return 0
            
        # Use neural network for evaluation
        state_tensor = self.get_state_tensor(game)
        with torch.no_grad():
            _, value = self.model(state_tensor)
        return value.item()

    def get_state_tensor(self, game):
        # Convert board to tensor
        board_tensor = torch.zeros((1, 3, BOARD_SIZE, BOARD_SIZE), device=self.device)
        
        # Channel 0: X pieces
        # Channel 1: O pieces
        # Channel 2: Valid moves
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if game.board[r][c] == PLAYER_X:
                    board_tensor[0, 0, r, c] = 1
                elif game.board[r][c] == PLAYER_O:
                    board_tensor[0, 1, r, c] = 1
                
        # Add valid moves to channel 2
        valid_moves = game.get_valid_moves()
        for r, c in valid_moves:
            board_tensor[0, 2, r, c] = 1
        
        return board_tensor

    def train(self, num_games=100):
        print(f"Starting training for {num_games} games...")
        wins = {'X': 0, 'O': 0, 'Draw': 0}
        total_moves = 0
        
        # Use tqdm for progress tracking
        from tqdm import tqdm
        pbar = tqdm(total=num_games, desc="Training Progress")
        
        for game_num in range(num_games):
            game = UltimateTicTacToe()
            states = []
            move_count = 0
            
            while not game.game_over:
                move_count += 1
                
                # Get current state and make move
                current_state = game.copy()
                move = self.mcts(game)
                
                if not move:
                    print(f"Game {game_num + 1}: No valid moves available")
                    break
                
                states.append((current_state, None))
                success = game.make_move(move[0], move[1])
                
                if not success:
                    print(f"Game {game_num + 1}: Invalid move detected")
                    break
            
            # Game finished, get result
            total_moves += move_count
            if game.winner == PLAYER_X:
                wins['X'] += 1
                result = 1.0
                print(f"\nGame {game_num + 1}/{num_games}: X wins in {move_count} moves!")
            elif game.winner == PLAYER_O:
                wins['O'] += 1
                result = -1.0
                print(f"\nGame {game_num + 1}/{num_games}: O wins in {move_count} moves!")
            else:
                wins['Draw'] += 1
                result = 0.0
                print(f"\nGame {game_num + 1}/{num_games}: Draw after {move_count} moves!")
            
            # Update states with actual game result
            for i, (state, _) in enumerate(states):
                value = result if state.current_player == PLAYER_X else -result
                states[i] = (state, value)
            
            # Add states to memory
            self.memory.extend(states)
            
            # Train on random batch from memory if we have enough samples
            if len(self.memory) >= self.batch_size:
                batch = random.sample(self.memory, self.batch_size)
                loss = self.train_batch(batch)
                print(f"Training loss: {loss:.4f}")
            
            # Print current statistics
            total = game_num + 1
            avg_moves = total_moves / total
            print(f"\nCurrent Statistics:")
            print(f"X Wins: {wins['X']}/{total} ({wins['X']/total*100:.1f}%)")
            print(f"O Wins: {wins['O']}/{total} ({wins['O']/total*100:.1f}%)")
            print(f"Draws: {wins['Draw']}/{total} ({wins['Draw']/total*100:.1f}%)")
            print(f"Average moves per game: {avg_moves:.1f}")
            print("-" * 50)
            
            # Update progress bar
            pbar.update(1)
        
        pbar.close()
        
        # Save the model
        torch.save(self.model.state_dict(), 'ultimate_tic_tac_toe_model.pth')
        print("\nTraining completed! Model saved successfully!")
        print(f"Final Statistics:")
        print(f"Total games: {num_games}")
        print(f"X Wins: {wins['X']} ({wins['X']/num_games*100:.1f}%)")
        print(f"O Wins: {wins['O']} ({wins['O']/num_games*100:.1f}%)")
        print(f"Draws: {wins['Draw']} ({wins['Draw']/num_games*100:.1f}%)")
        print(f"Average moves per game: {total_moves/num_games:.1f}")

    def train_batch(self, batch):
        # Vectorize state tensor creation
        states = torch.cat([self.get_state_tensor(game) for game, _ in batch])
        target_values = torch.tensor([value for _, value in batch], dtype=torch.float32, device=self.device)
        
        self.optimizer.zero_grad()
        _, values = self.model(states)
        loss = F.mse_loss(values.squeeze(), target_values)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

if __name__ == "__main__":
    # Check available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Initialize model with optimized parameters
    alpha_zero = AlphaZero(device=device)
    alpha_zero.mcts_iterations = 200  # Reduced for faster training
    alpha_zero.batch_size = 32  # Reduced for better GPU memory usage
    alpha_zero.temperature = 1.0
    alpha_zero.alpha = 0.3
    
    # Train model
    try:
        alpha_zero.train(num_games=200)
        print(f"Model saved successfully and can be loaded on {device}")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e 