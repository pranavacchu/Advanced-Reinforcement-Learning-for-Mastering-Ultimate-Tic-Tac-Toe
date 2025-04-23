from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
from alpha_zero import AlphaZero, AlphaZeroNet
import numpy as np
import json
import random
import os
from fastapi.responses import JSONResponse
from fastapi import Request

# Add the improved model architecture classes
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
        query = self.query(x).view(batch_size, -1, height * width)
        key = self.key(x).view(batch_size, -1, height * width)
        attention = torch.bmm(query.transpose(1, 2), key)
        attention = F.softmax(attention, dim=2)
        value = self.value(x).view(batch_size, -1, height * width)
        out = torch.bmm(value, attention.transpose(1, 2))
        out = out.view(batch_size, channels, height, width)
        return self.gamma * out + x

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

app = FastAPI()

# Enable CORS with more permissive settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # More permissive for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Add explicit OPTIONS handler
@app.options("/make-move")
async def options_handler():
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )

# Initialize game state
game_state = {
    "board": [[0 for _ in range(9)] for _ in range(9)],
    "meta_board": [[0 for _ in range(3)] for _ in range(3)],
    "current_player": 1,
    "active_sub_row": None,
    "active_sub_col": None,
    "last_move": None,
    "winner": None,
    "game_over": False
}

# Load AI models
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load hard difficulty model
    hard_model = AlphaZero(device=device)
    hard_model_path = 'ultimate_tic_tac_toe_model.pth'
    if os.path.exists(hard_model_path):
        state_dict = torch.load(hard_model_path, map_location=device)
        hard_model.model.load_state_dict(state_dict)
        print(f"Hard difficulty model loaded successfully on {device}")
    else:
        print("No hard difficulty model found. Using untrained model.")
    
    # Load medium difficulty model with improved architecture
    medium_model = ImprovedDQN().to(device)
    medium_model_path = 'final_improved_model.pt'
    if os.path.exists(medium_model_path):
        checkpoint = torch.load(medium_model_path, map_location=device)
        if 'policy_net' in checkpoint:
            # Load from training checkpoint format
            medium_model.load_state_dict(checkpoint['policy_net'])
        else:
            # Load from direct state dict format
            medium_model.load_state_dict(checkpoint)
        medium_model.eval()  # Set to evaluation mode
        print(f"Medium difficulty model loaded successfully on {device}")
    else:
        print("No medium difficulty model found.")
        medium_model = None
except Exception as e:
    print(f"Error loading AI models: {str(e)}")
    hard_model = None
    medium_model = None

@app.get("/")
async def root():
    return {"message": "Ultimate Tic Tac Toe API"}

@app.get("/game-state")
async def get_game_state():
    return game_state

@app.post("/reset")
async def reset_game():
    global game_state
    game_state = {
        "board": [[0 for _ in range(9)] for _ in range(9)],
        "meta_board": [[0 for _ in range(3)] for _ in range(3)],
        "current_player": 1,
        "active_sub_row": None,
        "active_sub_col": None,
        "last_move": None,
        "winner": None,
        "game_over": False
    }
    return game_state

@app.post("/make-move")
async def make_move(request: Request):
    try:
        move = await request.json()
        row = move.get("row")
        col = move.get("col")
        player = move.get("player")
        difficulty = move.get("difficulty", "easy")

        if not isinstance(row, int) or not isinstance(col, int):
            raise HTTPException(status_code=400, detail="Invalid move coordinates type")

        if not (0 <= row < 9 and 0 <= col < 9):
            raise HTTPException(status_code=400, detail="Invalid move coordinates range")

        if game_state["game_over"]:
            raise HTTPException(status_code=400, detail="Game is already over")

        if game_state["board"][row][col] != 0:
            raise HTTPException(status_code=400, detail="Cell already occupied")

        # Validate move based on active sub-board
        sub_row, sub_col = row // 3, col // 3
        if game_state["active_sub_row"] is not None and game_state["active_sub_col"] is not None:
            if sub_row != game_state["active_sub_row"] or sub_col != game_state["active_sub_col"]:
                if not is_sub_board_playable(game_state, game_state["active_sub_row"], game_state["active_sub_col"]):
                    if not is_sub_board_playable(game_state, sub_row, sub_col):
                        raise HTTPException(status_code=400, detail="Selected sub-board is not playable")
                else:
                    raise HTTPException(status_code=400, detail="Must play in the active sub-board")

        # Make player move
        game_state["board"][row][col] = player
        game_state["last_move"] = (row, col)
        
        # Update meta-board
        game_state["meta_board"][sub_row][sub_col] = check_sub_board_winner(
            game_state["board"], sub_row, sub_col
        )
        
        # Update active sub-board for next move
        next_sub_row, next_sub_col = row % 3, col % 3
        if is_sub_board_playable(game_state, next_sub_row, next_sub_col):
            game_state["active_sub_row"] = next_sub_row
            game_state["active_sub_col"] = next_sub_col
        else:
            game_state["active_sub_row"] = None
            game_state["active_sub_col"] = None
        
        # Check for winner
        game_state["winner"] = check_global_winner(game_state["meta_board"])
        if game_state["winner"] is not None:
            game_state["game_over"] = True
        elif all(cell != 0 for row in game_state["board"] for cell in row):
            game_state["game_over"] = True
        
        # If game is not over and it's AI's turn, make AI move
        if not game_state["game_over"] and player == 1:
            valid_moves = get_valid_moves(game_state)
            if valid_moves:
                if difficulty == "hard" and hard_model:
                    try:
                        ai_move = hard_model.mcts(game_state)
                        if not ai_move or ai_move not in valid_moves:
                            ai_move = random.choice(valid_moves)
                    except:
                        ai_move = random.choice(valid_moves)
                elif difficulty == "medium" and medium_model:
                    try:
                        # Prepare state for improved model
                        board_tensor = prepare_state_for_improved_model(game_state)
                        with torch.no_grad():
                            policy, _ = medium_model(board_tensor)
                            # Get move probabilities and mask invalid moves
                            move_probs = F.softmax(policy, dim=1).squeeze()
                            valid_move_mask = torch.zeros_like(move_probs)
                            valid_move_mask[valid_moves] = 1
                            masked_probs = move_probs * valid_move_mask
                            if masked_probs.sum() > 0:
                                ai_move = valid_moves[torch.argmax(masked_probs).item()]
                            else:
                                ai_move = random.choice(valid_moves)
                    except Exception as e:
                        print(f"Medium model error: {str(e)}")
                        ai_move = random.choice(valid_moves)
                else:  # Easy difficulty
                    ai_move = random.choice(valid_moves)

                game_state["board"][ai_move[0]][ai_move[1]] = -1
                game_state["last_move"] = ai_move
                
                # Update meta-board after AI move
                ai_sub_row, ai_sub_col = ai_move[0] // 3, ai_move[1] // 3
                game_state["meta_board"][ai_sub_row][ai_sub_col] = check_sub_board_winner(
                    game_state["board"], ai_sub_row, ai_sub_col
                )
                
                # Update active sub-board for next move
                next_sub_row, next_sub_col = ai_move[0] % 3, ai_move[1] % 3
                if is_sub_board_playable(game_state, next_sub_row, next_sub_col):
                    game_state["active_sub_row"] = next_sub_row
                    game_state["active_sub_col"] = next_sub_col
                else:
                    game_state["active_sub_row"] = None
                    game_state["active_sub_col"] = None
                
                # Check for winner after AI move
                game_state["winner"] = check_global_winner(game_state["meta_board"])
                if game_state["winner"] is not None:
                    game_state["game_over"] = True

        return game_state
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing move: {str(e)}")

def check_sub_board_winner(board, sub_row, sub_col):
    # Get the sub-board boundaries
    start_row = sub_row * 3
    start_col = sub_col * 3
    
    # Extract the sub-board
    sub_board = [
        [board[start_row + i][start_col + j] for j in range(3)]
        for i in range(3)
    ]
    
    # Check rows
    for row in sub_board:
        if all(cell == 1 for cell in row):
            return 1
        if all(cell == -1 for cell in row):
            return -1
    
    # Check columns
    for col in range(3):
        if all(sub_board[row][col] == 1 for row in range(3)):
            return 1
        if all(sub_board[row][col] == -1 for row in range(3)):
            return -1
    
    # Check diagonals
    if all(sub_board[i][i] == 1 for i in range(3)) or all(sub_board[i][2-i] == 1 for i in range(3)):
        return 1
    if all(sub_board[i][i] == -1 for i in range(3)) or all(sub_board[i][2-i] == -1 for i in range(3)):
        return -1
    
    return 0

def check_global_winner(meta_board):
    # Check rows and columns
    for i in range(3):
        if all(cell == 1 for cell in meta_board[i]) or all(row[i] == 1 for row in meta_board):
            return 1
        if all(cell == -1 for cell in meta_board[i]) or all(row[i] == -1 for row in meta_board):
            return -1
    
    # Check diagonals
    if all(meta_board[i][i] == 1 for i in range(3)) or all(meta_board[i][2-i] == 1 for i in range(3)):
        return 1
    if all(meta_board[i][i] == -1 for i in range(3)) or all(meta_board[i][2-i] == -1 for i in range(3)):
        return -1
    
    return None

def is_sub_board_playable(game_state, sub_row, sub_col):
    # Only check if the sub-board has empty spaces
    start_row = sub_row * 3
    start_col = sub_col * 3
    for r in range(start_row, start_row + 3):
        for c in range(start_col, start_col + 3):
            if game_state["board"][r][c] == 0:
                return True
    return False

def get_valid_moves(game_state):
    valid_moves = []
    if game_state["active_sub_row"] is None or game_state["active_sub_col"] is None:
        # Can play in any open cell in any playable sub-board
        for sub_row in range(3):
            for sub_col in range(3):
                if is_sub_board_playable(game_state, sub_row, sub_col):
                    start_row = sub_row * 3
                    start_col = sub_col * 3
                    for r in range(start_row, start_row + 3):
                        for c in range(start_col, start_col + 3):
                            if game_state["board"][r][c] == 0:
                                valid_moves.append((r, c))
    else:
        # Must play in the active sub-board if it's playable
        start_row = game_state["active_sub_row"] * 3
        start_col = game_state["active_sub_col"] * 3
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                if game_state["board"][r][c] == 0:
                    valid_moves.append((r, c))
        
        # If active sub-board is not playable, can play in any open cell in any playable sub-board
        if not valid_moves or not is_sub_board_playable(game_state, game_state["active_sub_row"], game_state["active_sub_col"]):
            for sub_row in range(3):
                for sub_col in range(3):
                    if is_sub_board_playable(game_state, sub_row, sub_col):
                        start_row = sub_row * 3
                        start_col = sub_col * 3
                        for r in range(start_row, start_row + 3):
                            for c in range(start_col, start_col + 3):
                                if game_state["board"][r][c] == 0:
                                    valid_moves.append((r, c))
    return valid_moves

def prepare_state_for_improved_model(state):
    """Convert game state to tensor format for improved model"""
    # Create 4 channels: player pieces, opponent pieces, meta board, and active sub-board
    channels = np.zeros((4, 9, 9), dtype=np.float32)
    
    # Player pieces (X)
    channels[0] = (np.array(state["board"]) == 1).astype(np.float32)
    # Opponent pieces (O)
    channels[1] = (np.array(state["board"]) == -1).astype(np.float32)
    
    # Meta board
    meta = np.array(state["meta_board"])
    for i in range(3):
        for j in range(3):
            if meta[i, j] != 0:
                channels[2, i*3:(i+1)*3, j*3:(j+1)*3] = meta[i, j]
    
    # Active sub-board
    if state["active_sub_row"] is not None and state["active_sub_col"] is not None:
        start_row = state["active_sub_row"] * 3
        start_col = state["active_sub_col"] * 3
        channels[3, start_row:start_row+3, start_col:start_col+3] = 1.0
    
    return torch.FloatTensor(channels).unsqueeze(0).to(device)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 