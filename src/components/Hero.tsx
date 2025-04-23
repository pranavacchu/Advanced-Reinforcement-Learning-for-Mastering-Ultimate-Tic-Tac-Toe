import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { ArrowRight, Zap } from "lucide-react";
import { useNavigate } from 'react-router-dom';
import RulesModal from './RulesModal';

const Hero = () => {
  const navigate = useNavigate();
  const [isRulesModalOpen, setIsRulesModalOpen] = useState(false);

  return (
    <div className="relative z-10 min-h-screen flex flex-col items-center justify-center text-center px-4 pt-24">
      <div className="space-y-6">
        <div className="overflow-hidden">
          <p className="text-tictac-blue-light font-medium uppercase tracking-wider animate-on-scroll transition-all duration-500 transform translate-y-0 mt-16">WELCOME TO ULTIMATE TIC-TAC-TOE</p>
        </div>
        
        <div className="overflow-hidden">
          <h1 className="text-5xl md:text-7xl lg:text-8xl font-bold tracking-tighter mt-4 mb-6 text-white animate-on-scroll transition-all duration-700 delay-200">
            Strategic<br/>
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-tictac-purple via-tictac-blue-light to-tictac-purple animate-gradient-shift">
              Tic-Tac-Toe.
            </span>
          </h1>
        </div>
        
        <div className="overflow-hidden">
          <p className="text-lg md:text-xl max-w-2xl mx-auto text-white/80 mb-8 animate-on-scroll transition-all duration-700 delay-300">
            A next-level experience of the classic game with advanced strategy,
            multiple boards, and competitive gameplay.
          </p>
        </div>
        
        <div className="flex flex-col sm:flex-row gap-6 justify-center animate-on-scroll transition-all duration-700 delay-500">
          <Button 
            className="bg-gradient-to-r from-tictac-blue to-tictac-blue-light text-white px-8 py-7 text-lg animated-button group"
            onClick={() => navigate('/play')}
          >
            Play Now 
            <ArrowRight className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform duration-300" />
          </Button>
          <Button 
            variant="outline" 
            className="bg-transparent border-white/20 hover:bg-white/10 text-white px-8 py-7 text-lg hover:border-tictac-purple/50 transition-all duration-300"
            onClick={() => setIsRulesModalOpen(true)}
          >
            Learn Rules
          </Button>
        </div>
      </div>
      
      <div className="mt-24 glass-effect p-6 rounded-lg max-w-md animate-on-scroll transition-all duration-700 delay-700 hover:border-tictac-purple/30 hover:scale-105 transition-transform">
        <div className="flex items-center mb-4">
          <div className="h-10 w-10 rounded-full bg-tictac-purple/20 flex items-center justify-center">
            <Zap className="h-5 w-5 text-tictac-purple" />
          </div>
          <p className="font-medium ml-3 text-lg text-white">Quick Stats</p>
        </div>
        <div className="grid grid-cols-3 gap-4 text-center">
          <div className="p-3 hover:bg-white/5 rounded-lg transition-colors">
            <p className="text-tictac-purple text-2xl font-bold mb-1 counter">10K+</p>
            <p className="text-xs text-white/70">Daily Games</p>
          </div>
          <div className="p-3 hover:bg-white/5 rounded-lg transition-colors">
            <p className="text-tictac-purple text-2xl font-bold mb-1">24/7</p>
            <p className="text-xs text-white/70">Available</p>
          </div>
          <div className="p-3 hover:bg-white/5 rounded-lg transition-colors">
            <p className="text-tictac-purple text-2xl font-bold mb-1 counter">1.2M</p>
            <p className="text-xs text-white/70">Players</p>
          </div>
        </div>
      </div>

      <RulesModal 
        isOpen={isRulesModalOpen} 
        onClose={() => setIsRulesModalOpen(false)} 
      />
    </div>
  );
};

export default Hero;
