# Ultimate Tic Tac Toe

![Ultimate Tic Tac Toe](https://img.shields.io/badge/Game-Ultimate%20Tic%20Tac%20Toe-blue)
![React](https://img.shields.io/badge/Frontend-React-61DAFB)
![TypeScript](https://img.shields.io/badge/Language-TypeScript-3178C6)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![PyTorch](https://img.shields.io/badge/AI-PyTorch-EE4C2C)
![Neural Networks](https://img.shields.io/badge/AI-Neural%20Networks-FF6F00)
![AlphaZero](https://img.shields.io/badge/AI-AlphaZero-76B900)

A modern implementation of Ultimate Tic Tac Toe with advanced AI opponents powered by neural networks, reinforcement learning, and AlphaZero algorithms.

![Game Demo](https://via.placeholder.com/800x400?text=Ultimate+Tic+Tac+Toe+Screenshot) <!-- Replace with an actual screenshot of your game -->

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies](#technologies)
- [Game Rules](#game-rules)
- [AI Implementation](#ai-implementation)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Development](#development)
- [Future Improvements](#future-improvements)
- [License](#license)

## Overview

Ultimate Tic Tac Toe is a complex variation of the classic Tic Tac Toe game, where each cell of the traditional 3x3 grid contains another 3x3 grid. This implementation offers a modern, visually appealing user interface with cutting-edge AI opponents that learn and adapt using advanced machine learning techniques.

The game features three difficulty levels, powered by distinct AI approaches:
- Easy: Strategic random moves
- Medium: Neural network with residual blocks and attention mechanisms
- Hard: AlphaZero-based AI using Monte Carlo Tree Search (MCTS) and reinforcement learning

## Features

- **Beautiful User Interface**: Modern UI with animations, floating elements, and responsive design
- **Three AI Difficulty Levels**: Choose between easy, medium, and hard computer opponents
- **Advanced AI**: Neural network and AlphaZero-based opponents
- **Real-time Game State**: Displays current player and game progress
- **Victory Celebrations**: Visual effects when winning the game
- **Game Rules Guide**: Integrated tutorial on how to play Ultimate Tic Tac Toe
- **Responsive Design**: Play on devices of any size

## Technologies

### Frontend
- **React**: UI library for building the game interface
- **TypeScript**: Type-safe JavaScript for robust code
- **Vite**: Next-generation frontend tooling for faster development
- **Lucide-React**: Beautiful icons for UI elements
- **React Router**: Handling navigation between pages
- **React Confetti**: Victory celebration effects
- **Custom CSS Animations**: For smooth visual transitions

### Backend
- **FastAPI**: High-performance Python web framework
- **PyTorch**: Machine learning framework for neural networks
- **NumPy**: Numerical computing for game state processing
- **Uvicorn**: ASGI server for running the FastAPI application
- **CORS Middleware**: Support for cross-origin requests

### AI Implementation
- **AlphaZero Algorithm**: Advanced game AI using neural networks and MCTS
- **Residual Neural Networks**: Deep learning model with residual connections
- **Attention Mechanisms**: Neural network attention for better decision making
- **Monte Carlo Tree Search**: Strategic game tree exploration

## Game Rules

Ultimate Tic Tac Toe adds complexity to the classic game:

1. The game board consists of a 3×3 grid of 3×3 Tic Tac Toe boards (9 smaller boards).
2. To win, you need to win three smaller boards in a row (horizontally, vertically, or diagonally).
3. Your move dictates where your opponent can play next:
   - When you place your mark in a small square, your opponent must play in the corresponding small board.
   - Example: If you play in the top-right square of any small board, your opponent must play somewhere in the top-right small board.
4. If the board your opponent is sent to is already won or full, they can play in any available space on any open board.

This creates a strategic depth where each move affects not just the current position, but future possibilities as well.

## AI Implementation

### Easy Mode
The easy difficulty uses strategic random selection from available valid moves, providing a good starting point for beginners.

### Medium Mode
Medium difficulty uses an improved deep neural network with:
- **Residual blocks**: For better gradient flow and deeper network training
- **Attention mechanisms**: To focus on important board areas
- **Separate policy and value heads**: For move probability and position evaluation
- **Four-channel state representation**: Capturing player pieces, opponent pieces, meta board status, and active sub-board information

### Hard Mode
Hard difficulty implements a version of AlphaZero that combines:
- **Neural network evaluation**: Estimating move probabilities and position values
- **Monte Carlo Tree Search**: Exploring possible game states efficiently
- **Self-play reinforcement learning**: Learning through thousands of games against itself
- **Advanced pruning techniques**: For efficient state space search

## Installation

### Prerequisites
- Node.js 16+ and npm
- Python 3.8+
- Git

### Frontend Setup
```bash
# Clone the repository
git clone https://github.com/your-username/ultimate-tic-tac-toe.git
cd ultimate-tic-tac-toe

# Install frontend dependencies
cd src
npm install

# Start the development server
npm run dev
```

### Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create a virtual environment (optional but recommended)
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the backend server
python main.py
```

The frontend will be available at `http://localhost:5173` and the backend API at `http://localhost:8000`.

## Project Structure

```
ultimate-tic-tac-toe/
├── src/                      # Frontend code
│   ├── components/           # Reusable UI components
│   │   ├── Features.tsx      # Game feature showcase
│   │   ├── FloatingSpheres.tsx # Background animation
│   │   ├── Footer.tsx        # Page footer
│   │   ├── Hero.tsx          # Landing page hero section
│   │   ├── Navbar.tsx        # Navigation bar
│   │   ├── RulesModal.tsx    # Game rules explanation
│   │   └── ui/               # UI component library
│   ├── hooks/                # Custom React hooks
│   ├── lib/                  # Utility functions
│   ├── pages/                # Application pages
│   │   ├── GameBoard.tsx     # Main game component
│   │   ├── Index.tsx         # Landing page
│   │   └── NotFound.tsx      # 404 page
│   ├── styles/               # CSS styles
│   ├── App.css               # Global styles
│   ├── App.tsx               # Root component
│   └── main.tsx              # Entry point
├── backend/                  # Backend code
│   ├── alpha_zero.py         # AlphaZero implementation
│   ├── improved_training.py  # Training script for AI models
│   ├── main.py               # FastAPI server
│   ├── ppo_train.py          # PPO reinforcement learning
│   ├── requirements.txt      # Python dependencies
│   ├── run.py                # Server runner
│   └── models/               # Neural network models
│       ├── final_improved_model.pt    # Medium difficulty model
│       └── ultimate_tic_tac_toe_model.pth # Hard difficulty model
└── README.md                 # Project documentation
```

## Development

### Frontend Development
```bash
# Run development server with hot-reload
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Backend Development
```bash
# Run FastAPI server with auto-reload
uvicorn main:app --reload

# Run tests
pytest
```

### AI Model Training
```bash
# Train improved neural network model
python improved_training.py

# Train AlphaZero model
python alpha_zero.py --train
```

## Future Improvements

- **Multiplayer Mode**: Online play against other players
- **Game Replay**: Record and replay past games
- **Customizable Themes**: Different visual styles for the game board
- **Advanced Statistics**: Track win rates and performance metrics
- **Tournament Mode**: Compete in a series of games with scoring
- **Mobile App**: Native mobile applications using React Native

## License

[MIT License](LICENSE) - Feel free to use, modify, and distribute this code for your own projects.

## Acknowledgements

- Deep learning algorithms based on research papers by DeepMind (AlphaZero)
- Game implementation inspired by ultimate tic-tac-toe competitive play
- UI design elements from modern web application trends

---

Created by [Your Name/Username] - [Your Website/GitHub](https://github.com/yourusername)
