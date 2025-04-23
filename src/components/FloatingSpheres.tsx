
import React, { useEffect, useRef } from 'react';

interface SphereProps {
  size: string;
  color: string;
  top: string;
  left: string;
  delay?: string;
  duration?: string;
  opacity?: string;
}

const Sphere: React.FC<SphereProps> = ({ 
  size, 
  color, 
  top, 
  left, 
  delay = '0s', 
  duration = '8s', 
  opacity = '0.7' 
}) => {
  return (
    <div 
      className="absolute rounded-full floating-sphere animate-float"
      style={{
        width: size,
        height: size,
        backgroundColor: color,
        boxShadow: `0 0 60px ${color}40`,
        top,
        left,
        opacity,
        animationDelay: delay,
        animationDuration: duration,
      }}
    />
  );
};

const FloatingSpheres: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Add parallax effect
    const handleMouseMove = (e: MouseEvent) => {
      if (!containerRef.current) return;
      
      const spheres = containerRef.current.querySelectorAll('.floating-sphere');
      const x = e.clientX / window.innerWidth;
      const y = e.clientY / window.innerHeight;
      
      spheres.forEach((sphere, i) => {
        const speed = 1 - (i % 3) * 0.1;
        const sphereElement = sphere as HTMLElement;
        const offsetX = (x - 0.5) * 20 * speed;
        const offsetY = (y - 0.5) * 20 * speed;
        
        sphereElement.style.transform = `translate(${offsetX}px, ${offsetY}px)`;
      });
    };
    
    window.addEventListener('mousemove', handleMouseMove);
    
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  return (
    <div ref={containerRef} className="fixed inset-0 overflow-hidden z-0">
      <Sphere size="130px" color="#9b87f5" top="15%" left="10%" delay="0s" opacity="0.4" duration="12s" />
      <Sphere size="90px" color="#0EA5E9" top="70%" left="20%" delay="0.5s" opacity="0.5" duration="10s" />
      <Sphere size="170px" color="#7E69AB" top="40%" left="80%" delay="1.2s" opacity="0.35" duration="14s" />
      <Sphere size="110px" color="#33C3F0" top="80%" left="85%" delay="0.8s" opacity="0.4" duration="11s" />
      <Sphere size="80px" color="#1EAEDB" top="25%" left="50%" delay="1.5s" opacity="0.6" duration="9s" />
      <Sphere size="100px" color="#8B5CF6" top="60%" left="40%" delay="2s" opacity="0.45" duration="13s" />
      <Sphere size="75px" color="#6366f1" top="10%" left="70%" delay="1s" opacity="0.5" duration="15s" />
      <Sphere size="120px" color="#3f3f50" top="85%" left="5%" delay="1.7s" opacity="0.25" duration="16s" />
      <Sphere size="95px" color="#D3E4FD" top="5%" left="30%" delay="0.3s" opacity="0.35" duration="14s" />
    </div>
  );
};

export default FloatingSpheres;
