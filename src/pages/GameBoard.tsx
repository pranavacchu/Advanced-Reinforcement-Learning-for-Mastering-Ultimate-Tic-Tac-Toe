import React, { useState, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { ArrowLeft, RotateCcw, Sparkles, Cpu, Brain, Zap, Trophy } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import FloatingSpheres from '@/components/FloatingSpheres';
import Confetti from 'react-confetti';
import '../styles/game.css';

const GameBoard = () => {
  const navigate = useNavigate();
  const [currentPlayer, setCurrentPlayer] = useState<'X' | 'O'>('X');
  const [selectedBoard, setSelectedBoard] = useState<number | null>(null);
  const [mainBoardState, setMainBoardState] = useState<Array<string>>(Array(9).fill(''));
  const [difficulty, setDifficulty] = useState<'easy' | 'medium' | 'hard'>('easy');
  const [isLoading, setIsLoading] = useState(false);
  const [winner, setWinner] = useState<'X' | 'O' | null>(null);
  const [showConfetti, setShowConfetti] = useState(false);

  // Initialize the game state for all boards
  const [gameState, setGameState] = useState(() => {
    const initialState = Array(9).fill(null).map(() => Array(9).fill(''));
    return initialState;
  });

  const handleCellClick = async (boardIndex: number, cellIndex: number) => {
    if (selectedBoard !== null && selectedBoard !== boardIndex) return;
    if (isLoading) return;
    if (winner) return;  // Prevent moves if there's a winner

    try {
      setIsLoading(true);
      const response = await fetch('http://localhost:8000/make-move', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          row: Math.floor(boardIndex / 3) * 3 + Math.floor(cellIndex / 3),
          col: (boardIndex % 3) * 3 + (cellIndex % 3),
          player: 1,
          difficulty: difficulty
        }),
      });

      if (!response.ok) {
        throw new Error('Invalid move');
      }

      const data = await response.json();
      updateGameStateFromResponse(data);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const updateGameStateFromResponse = (data: any) => {
    // Convert the 1D board to our 9x9 format
    const newGameState = Array(9).fill(null).map(() => Array(9).fill(''));
    for (let i = 0; i < 9; i++) {
      for (let j = 0; j < 9; j++) {
        const value = data.board[i][j];
        newGameState[Math.floor(i / 3) * 3 + Math.floor(j / 3)][(i % 3) * 3 + j % 3] = 
          value === 1 ? 'X' : value === -1 ? 'O' : '';
      }
    }
    setGameState(newGameState);

    // Update main board state
    const newMainBoard = data.meta_board.flat().map((value: number) => 
      value === 1 ? 'X' : value === -1 ? 'O' : ''
    );
    setMainBoardState(newMainBoard);

    // Update current player
    setCurrentPlayer(data.current_player === 1 ? 'X' : 'O');

    // Update selected board based on active sub-board
    setSelectedBoard(
      data.active_sub_row !== null && data.active_sub_col !== null
        ? data.active_sub_row * 3 + data.active_sub_col
        : null
    );

    // Check for winner
    if (data.winner !== null) {
      const gameWinner = data.winner === 1 ? 'X' : 'O';
      setWinner(gameWinner);
      setShowConfetti(true);
      // Hide confetti after 5 seconds
      setTimeout(() => setShowConfetti(false), 5000);
    }
  };

  const resetGame = async () => {
    try {
      setIsLoading(true);
      setWinner(null);  // Reset winner state
      setShowConfetti(false);  // Hide confetti
      const response = await fetch('http://localhost:8000/reset', {
        method: 'POST',
      });
      const data = await response.json();
      updateGameStateFromResponse(data);
    } catch (error) {
      console.error('Error resetting game:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const renderMainBoard = () => {
    return (
      <div className="bg-black/30 p-4 rounded-xl backdrop-blur-sm">
        <h3 className="text-white/80 text-lg mb-3 font-medium text-center">Game Progress</h3>
        <div className="grid grid-cols-3 gap-2 w-48">
          {mainBoardState.map((value, index) => (
            <div key={index}
              className={`aspect-square flex items-center justify-center text-3xl font-bold
                ${value === 'X' ? 'text-tictac-blue-light' : 'text-tictac-purple'}
                ${value ? 'bg-black/40' : 'bg-black/20'}
                rounded-lg transition-all duration-300 border border-white/20`}
            >
              {value}
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderCell = (boardIndex: number, cellIndex: number) => {
    const value = gameState[boardIndex][cellIndex];
    const isPlayable = selectedBoard === null || selectedBoard === boardIndex;

    return (
      <button
        className={`w-full h-full flex items-center justify-center text-3xl font-bold transition-all duration-300
          ${value === 'X' ? 'text-tictac-blue-light' : 'text-tictac-purple'}
          ${!value && isPlayable ? 'hover:bg-white/10' : ''}
          ${!isPlayable ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
          font-game tracking-wide rounded-md
        `}
        onClick={() => handleCellClick(boardIndex, cellIndex)}
        disabled={!isPlayable || !!value}
      >
        {value}
      </button>
    );
  };

  const renderBoard = (boardIndex: number) => {
    const isPlayable = selectedBoard === null || selectedBoard === boardIndex;

    return (
      <div className={`relative backdrop-blur-sm rounded-lg p-3 transition-all duration-300
        ${isPlayable ? 'ring-2 ring-white/20 hover:ring-tictac-purple/50 shadow-lg shadow-tictac-purple/20' : 'opacity-75'}
        bg-gradient-to-br from-black/60 to-black/40
      `}>
        <div className="grid grid-cols-3 gap-2">
          {Array(9).fill(null).map((_, cellIndex) => (
            <div key={cellIndex}
              className="aspect-square bg-gradient-to-br from-white/10 to-transparent rounded-lg
                border-t border-l border-white/20 backdrop-blur-sm"
            >
              {renderCell(boardIndex, cellIndex)}
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderWinMessage = () => {
    if (!winner) return null;
    
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-md">
        <div className="win-message-container p-10 rounded-2xl text-center transform transition-all duration-500 win-message">
          <div className="win-message-trophy">
            <Trophy className={`w-20 h-20 mx-auto mb-6 ${winner === 'X' ? 'text-tictac-blue-light' : 'text-tictac-purple'}`} />
          </div>
          <h2 className={`text-5xl font-bold mb-4 font-game win-message-text`}>
            Player {winner} Wins!
          </h2>
          <p className="text-white/80 mb-8 text-lg">
            Congratulations on your victory!
          </p>
          <div className="flex justify-center gap-4">
            <Button
              variant="outline"
              className="win-action-button"
              onClick={resetGame}
            >
              Play Again
            </Button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-[#070711] relative overflow-hidden">
      {showConfetti && <Confetti
        width={window.innerWidth}
        height={window.innerHeight}
        recycle={false}
        numberOfPieces={500}
      />}
      {/* Background gradient */}
      <div className="fixed inset-0 z-0 gradient-bg opacity-10"></div>

      {/* Floating spheres with reduced opacity */}
      <div className="fixed inset-0 z-[1] opacity-30">
        <FloatingSpheres />
      </div>

      {renderWinMessage()}

      {/* Game content with higher z-index and glass effect */}
      <div className="relative z-10 container mx-auto px-4 py-8">
        <div className="header-container">
          <Button
            variant="ghost"
            className="back-button"
            onClick={() => navigate('/')}
          >
            <ArrowLeft />
          </Button>

          <div className="flex items-center gap-4">
            <div className="difficulty-selector">
              <button
                className={`difficulty-button ${difficulty === 'easy' ? 'active' : ''}`}
                onClick={() => {
                  setDifficulty('easy');
                  resetGame();
                }}
                disabled={isLoading}
              >
                <Brain size={20} />
                <span>Easy</span>
              </button>
              <button
                className={`difficulty-button ${difficulty === 'medium' ? 'active' : ''}`}
                onClick={() => {
                  setDifficulty('medium');
                  resetGame();
                }}
                disabled={isLoading}
              >
                <Cpu size={20} />
                <span>Medium</span>
              </button>
              <button
                className={`difficulty-button ${difficulty === 'hard' ? 'active' : ''}`}
                onClick={() => {
                  setDifficulty('hard');
                  resetGame();
                }}
                disabled={isLoading}
              >
                <Zap size={20} />
                <span>Hard</span>
              </button>
            </div>

            <Button 
              variant="outline" 
              className="reset-button"
              onClick={resetGame}
              disabled={isLoading}
            >
              <RotateCcw />
              Reset Game
            </Button>
          </div>
        </div>
        
        <div className="title-container">
          <h1 className="game-title">
            Ultimate Tic-Tac-Toe
          </h1>
          <div className={`turn-indicator ${currentPlayer === 'X' ? 'turn-x' : 'turn-o'}`}>
            Current Turn: {currentPlayer}
          </div>
        </div>
        
        {/* Game Layout */}
        <div className="flex justify-between items-start max-w-5xl mx-auto">
          {/* Main Game Board */}
          <div className="flex-1 mr-6">
            <div className="grid grid-cols-3 gap-3 md:gap-4 p-4 rounded-xl 
              bg-gradient-to-br from-black/50 to-black/30 backdrop-blur-md shadow-xl 
              border border-white/10 transform scale-90">
              {Array(9).fill(null).map((_, boardIndex) => (
                <div key={boardIndex} className="transform transition-all duration-300 hover:scale-[1.02]">
                  {renderBoard(boardIndex)}
                </div>
              ))}
            </div>
          </div>

          {/* Side Panel with Score */}
          <div className="w-56 sticky top-8">
            {renderMainBoard()}
          </div>
        </div>
      </div>
    </div>
  );
};

export default GameBoard; 