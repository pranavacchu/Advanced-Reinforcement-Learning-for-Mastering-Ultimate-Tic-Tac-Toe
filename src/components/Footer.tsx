import React from 'react';
import { Zap, Github, Twitter, Instagram } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="relative z-10 px-6 py-20 border-t border-white/5">
      <div className="max-w-6xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12 md:gap-8">
          <div className="md:col-span-1 animate-on-scroll">
            <div className="flex items-center space-x-2 mb-6">
              <Zap className="w-6 h-6 text-tictac-purple" />
              <span className="font-bold tracking-tighter text-xl text-white">ULTIMATE TIC-TAC-TOE</span>
            </div>
            <p className="text-white/90 text-sm leading-relaxed">
              A next-level experience of the classic game with advanced strategy and competitive gameplay.
            </p>
          </div>
          
          <div className="animate-on-scroll" style={{ animationDelay: "100ms" }}>
            <h3 className="font-medium mb-6 text-lg text-white">Play</h3>
            <ul className="space-y-4 text-white/90">
              <li><a href="#" className="hover:text-tictac-purple transition-colors duration-300 inline-block after:content-[''] after:block after:w-0 after:h-0.5 after:bg-tictac-purple hover:after:w-full after:transition-all">Single Player</a></li>
              <li><a href="#" className="hover:text-tictac-purple transition-colors duration-300 inline-block after:content-[''] after:block after:w-0 after:h-0.5 after:bg-tictac-purple hover:after:w-full after:transition-all">Multiplayer</a></li>
              <li><a href="#" className="hover:text-tictac-purple transition-colors duration-300 inline-block after:content-[''] after:block after:w-0 after:h-0.5 after:bg-tictac-purple hover:after:w-full after:transition-all">Tournaments</a></li>
            </ul>
          </div>
          
          <div className="animate-on-scroll" style={{ animationDelay: "200ms" }}>
            <h3 className="font-medium mb-6 text-lg text-white">Learn</h3>
            <ul className="space-y-4 text-white/90">
              <li><a href="#" className="hover:text-tictac-purple transition-colors duration-300 inline-block after:content-[''] after:block after:w-0 after:h-0.5 after:bg-tictac-purple hover:after:w-full after:transition-all">How to Play</a></li>
              <li><a href="#" className="hover:text-tictac-purple transition-colors duration-300 inline-block after:content-[''] after:block after:w-0 after:h-0.5 after:bg-tictac-purple hover:after:w-full after:transition-all">Strategy Guide</a></li>
              <li><a href="#" className="hover:text-tictac-purple transition-colors duration-300 inline-block after:content-[''] after:block after:w-0 after:h-0.5 after:bg-tictac-purple hover:after:w-full after:transition-all">FAQ</a></li>
            </ul>
          </div>
          
          <div className="animate-on-scroll" style={{ animationDelay: "300ms" }}>
            <h3 className="font-medium mb-6 text-lg text-white">Connect</h3>
            <div className="flex space-x-6">
              <a href="#" className="text-white/90 hover:text-tictac-purple transition-colors duration-300 hover:scale-110 transform">
                <Github className="h-6 w-6" />
              </a>
              <a href="#" className="text-white/90 hover:text-tictac-purple transition-colors duration-300 hover:scale-110 transform">
                <Twitter className="h-6 w-6" />
              </a>
              <a href="#" className="text-white/90 hover:text-tictac-purple transition-colors duration-300 hover:scale-110 transform">
                <Instagram className="h-6 w-6" />
              </a>
            </div>
          </div>
        </div>
        
        <div className="mt-16 pt-8 border-t border-white/5 text-center text-white/80 text-sm animate-on-scroll" style={{ animationDelay: "400ms" }}>
          <p>© {new Date().getFullYear()} Ultimate Tic-Tac-Toe. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
