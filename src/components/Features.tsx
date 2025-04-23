
import React from 'react';
import { 
  Trophy, 
  Zap, 
  Users, 
  Brain, 
  Layers 
} from "lucide-react";

interface FeatureProps {
  title: string;
  description: string;
  icon: React.ReactNode;
  delay: string;
}

const Feature: React.FC<FeatureProps> = ({ title, description, icon, delay }) => {
  return (
    <div 
      className="glass-effect p-8 rounded-xl hover:border-tictac-purple/30 transition-all duration-500 animate-on-scroll hover:transform hover:scale-105"
      style={{ transitionDelay: delay }}
    >
      <div className="h-14 w-14 rounded-full bg-gradient-to-br from-tictac-purple/30 to-tictac-blue/30 flex items-center justify-center mb-6">
        {icon}
      </div>
      <h3 className="text-2xl font-bold mb-4 text-gradient">{title}</h3>
      <p className="text-white/70 leading-relaxed">{description}</p>
    </div>
  );
};

const Features = () => {
  return (
    <div className="relative z-10 py-32 px-6 md:px-12" id="explore">
      <div className="text-center mb-24 animate-on-scroll">
        <h2 className="text-4xl md:text-5xl font-bold mb-6 text-gradient">Game Features</h2>
        <p className="text-white/70 max-w-2xl mx-auto text-lg">
          Ultimate Tic-Tac-Toe reimagines the classic game with strategic depth
          and competitive elements for the modern player
        </p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-12 max-w-6xl mx-auto">
        <Feature 
          title="Strategic Depth" 
          description="Play on nine boards simultaneously with nested gameplay that requires advanced planning and tactics."
          icon={<Brain className="h-7 w-7 text-tictac-purple" />}
          delay="100ms"
        />
        
        <Feature 
          title="Competitive Rankings" 
          description="Climb the global leaderboards and earn achievements as you master the game and defeat opponents."
          icon={<Trophy className="h-7 w-7 text-tictac-purple" />}
          delay="200ms"
        />
        
        <Feature 
          title="Multiplayer" 
          description="Challenge friends or play against opponents of your skill level from around the world."
          icon={<Users className="h-7 w-7 text-tictac-purple" />}
          delay="300ms"
        />
        
        <Feature 
          title="Speed Modes" 
          description="Test your skills in timed matches that require quick thinking and faster decisions under pressure."
          icon={<Zap className="h-7 w-7 text-tictac-purple" />}
          delay="400ms"
        />
        
        <Feature 
          title="Customizable Experience" 
          description="Choose from various themes, board layouts, and gameplay modes to personalize your experience."
          icon={<Layers className="h-7 w-7 text-tictac-purple" />}
          delay="500ms"
        />
      </div>
    </div>
  );
};

export default Features;
