import React, { useEffect } from 'react';
import Navbar from '@/components/Navbar';
import FloatingSpheres from '@/components/FloatingSpheres';
import Hero from '@/components/Hero';
import Features from '@/components/Features';
import Footer from '@/components/Footer';

const Index = () => {
  // Add a scroll animation effect for elements
  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('animate-fade-in');
          entry.target.classList.remove('opacity-0');
        }
      });
    }, { threshold: 0.1 });

    document.querySelectorAll('.animate-on-scroll').forEach(el => {
      el.classList.add('opacity-0');
      observer.observe(el);
    });

    return () => {
      document.querySelectorAll('.animate-on-scroll').forEach(el => {
        observer.unobserve(el);
      });
    };
  }, []);

  return (
    <div className="bg-black min-h-screen overflow-hidden relative">
      {/* Dynamic gradient background */}
      <div className="fixed inset-0 z-0 gradient-bg opacity-20"></div>
      
      {/* Moving spheres in the background */}
      <FloatingSpheres />
      
      {/* Main content */}
      <div className="relative z-10">
        <Navbar />
        
        <main className="container mx-auto">
          <Hero />
          <Features />
        </main>
        
        <Footer />
      </div>
    </div>
  );
};

export default Index;