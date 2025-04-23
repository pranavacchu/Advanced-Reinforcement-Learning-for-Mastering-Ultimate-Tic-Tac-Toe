import React, { useEffect, useState } from 'react';
import { X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface RulesModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const RulesModal: React.FC<RulesModalProps> = ({ isOpen, onClose }) => {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    }
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isOpen]);

  if (!mounted) return null;

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
          className="fixed inset-0 z-50"
        >
          {/* Backdrop */}
          <motion.div
            className="absolute inset-0 bg-[rgba(13,12,34,0.95)] backdrop-blur-md"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />

          {/* Modal Content */}
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6"
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            transition={{ type: "spring", duration: 0.5 }}
          >
            <div 
              className="relative bg-gradient-to-br from-[rgba(28,27,60,0.98)] to-[rgba(18,17,40,0.98)] rounded-2xl w-full max-w-4xl max-h-[85vh] overflow-hidden shadow-2xl border border-white/10"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Header */}
              <div className="flex justify-between items-center p-6 border-b border-white/10 bg-white/5">
                <motion.h2 
                  className="text-3xl font-bold bg-gradient-to-r from-tictac-blue via-tictac-purple to-tictac-blue bg-clip-text text-transparent"
                  initial={{ opacity: 0, y: -20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                >
                  Ultimate Tic-Tac-Toe Rules
                </motion.h2>
                <motion.button 
                  onClick={onClose}
                  className="text-white/80 hover:text-white bg-white/5 hover:bg-white/10 rounded-full p-2 transition-all duration-300 hover:rotate-90 focus:outline-none focus:ring-2 focus:ring-tictac-blue/50"
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                >
                  <X size={24} />
                </motion.button>
              </div>

              {/* Scrollable Content */}
              <div className="p-6 overflow-y-auto max-h-[calc(85vh-80px)] custom-scrollbar">
                <div className="space-y-6">
                  <motion.div 
                    className="rule-section bg-white/5 p-6 rounded-xl border border-white/10 hover:border-tictac-blue/30 transition-all duration-300"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                    whileHover={{ scale: 1.02, transition: { duration: 0.2 } }}
                  >
                    <h3 className="text-tictac-blue text-xl font-semibold mb-3">Basic Concept</h3>
                    <p className="text-white/90 leading-relaxed">
                      Ultimate Tic-Tac-Toe is played on a 3×3 grid of 3×3 Tic-Tac-Toe boards. Win three games in a row to win the ultimate game!
                    </p>
                  </motion.div>

                  <motion.div 
                    className="rule-section bg-white/5 p-6 rounded-xl border border-white/10 hover:border-tictac-blue/30 transition-all duration-300"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                    whileHover={{ scale: 1.02, transition: { duration: 0.2 } }}
                  >
                    <h3 className="text-tictac-blue text-xl font-semibold mb-3">How to Play</h3>
                    <ol className="text-white/90 list-decimal pl-5 space-y-3">
                      <li className="hover:text-white transition-colors">Your move in one square determines which board your opponent plays next</li>
                      <li className="hover:text-white transition-colors">Win a small board by getting three in a row</li>
                      <li className="hover:text-white transition-colors">Win the game by winning three small boards in a row</li>
                    </ol>
                  </motion.div>

                  <motion.div 
                    className="rule-section bg-white/5 p-6 rounded-xl border border-white/10 hover:border-tictac-blue/30 transition-all duration-300"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 }}
                    whileHover={{ scale: 1.02, transition: { duration: 0.2 } }}
                  >
                    <h3 className="text-tictac-blue text-xl font-semibold mb-3">Special Rules</h3>
                    <ul className="text-white/90 list-disc pl-5 space-y-3">
                      <li className="hover:text-white transition-colors">If you're sent to a board that's already won,play in any left-over space in that board</li>
                      <li className="hover:text-white transition-colors">You can't play in a board that's full, you can play anywhere if you're forced to play in a full board</li>
                      <li className="hover:text-white transition-colors">If there's a tie in a small board, no one gets that square</li>
                    </ul>
                  </motion.div>

                  <motion.div 
                    className="rule-section bg-white/5 p-6 rounded-xl border border-white/10 hover:border-tictac-blue/30 transition-all duration-300"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.6 }}
                    whileHover={{ scale: 1.02, transition: { duration: 0.2 } }}
                  >
                    <h3 className="text-tictac-blue text-xl font-semibold mb-3">Strategy Tips</h3>
                    <ul className="text-white/90 list-disc pl-5 space-y-3">
                      <li className="hover:text-white transition-colors">Think ahead - your move determines where your opponent plays</li>
                      <li className="hover:text-white transition-colors">Control the center board when possible</li>
                      <li className="hover:text-white transition-colors">Watch for opportunities to create multiple winning paths</li>
                    </ul>
                  </motion.div>
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default RulesModal; 