import React from 'react'

const CosmicGif: React.FC = () => {
  return (
    <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
      {/* Animated cosmic background that simulates a space GIF */}
      
      {/* Moving stars field */}
      <div className="absolute inset-0">
        {Array.from({ length: 200 }).map((_, i) => (
          <div
            key={`star-${i}`}
            className="absolute w-1 h-1 bg-white rounded-full opacity-60"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationName: 'starMove',
              animationDuration: `${Math.random() * 20 + 10}s`,
              animationDelay: `${Math.random() * 20}s`,
              animationIterationCount: 'infinite',
              animationTimingFunction: 'linear'
            }}
          />
        ))}
      </div>
      
      {/* Nebula clouds */}
      <div className="absolute inset-0">
        {Array.from({ length: 8 }).map((_, i) => (
          <div
            key={`nebula-${i}`}
            className="absolute rounded-full opacity-20 blur-3xl"
            style={{
              width: `${Math.random() * 400 + 200}px`,
              height: `${Math.random() * 400 + 200}px`,
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              background: `radial-gradient(circle, ${
                ['#0B3D91', '#00FFFF', '#533a7b', '#16213e'][Math.floor(Math.random() * 4)]
              }40 0%, transparent 70%)`,
              animationName: 'nebulaDrift',
              animationDuration: `${Math.random() * 30 + 20}s`,
              animationDelay: `${Math.random() * 10}s`,
              animationIterationCount: 'infinite',
              animationTimingFunction: 'ease-in-out'
            }}
          />
        ))}
      </div>
      
      {/* Shooting stars */}
      <div className="absolute inset-0">
        {Array.from({ length: 5 }).map((_, i) => (
          <div
            key={`shooting-${i}`}
            className="absolute w-2 h-0.5 bg-gradient-to-r from-cyan-400 to-transparent opacity-80"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              transform: 'rotate(45deg)',
              animationName: 'shootingStar',
              animationDuration: '3s',
              animationDelay: `${Math.random() * 15}s`,
              animationIterationCount: 'infinite',
              animationTimingFunction: 'ease-out'
            }}
          />
        ))}
      </div>
      
      {/* Pulsing distant galaxies */}
      <div className="absolute inset-0">
        {Array.from({ length: 12 }).map((_, i) => (
          <div
            key={`galaxy-${i}`}
            className="absolute w-3 h-3 rounded-full opacity-40"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              background: `radial-gradient(circle, ${
                ['#00FFFF', '#0B3D91', '#FF006E', '#39FF14'][Math.floor(Math.random() * 4)]
              } 0%, transparent 100%)`,
              boxShadow: `0 0 20px ${
                ['#00FFFF', '#0B3D91', '#FF006E', '#39FF14'][Math.floor(Math.random() * 4)]
              }`,
              animationName: 'galaxyPulse',
              animationDuration: `${Math.random() * 4 + 2}s`,
              animationDelay: `${Math.random() * 5}s`,
              animationIterationCount: 'infinite',
              animationTimingFunction: 'ease-in-out'
            }}
          />
        ))}
      </div>
      
      {/* Cosmic dust particles */}
      <div className="absolute inset-0">
        {Array.from({ length: 100 }).map((_, i) => (
          <div
            key={`dust-${i}`}
            className="absolute w-0.5 h-0.5 bg-blue-200 rounded-full opacity-30"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationName: 'dustFloat',
              animationDuration: `${Math.random() * 25 + 15}s`,
              animationDelay: `${Math.random() * 10}s`,
              animationIterationCount: 'infinite',
              animationTimingFunction: 'linear'
            }}
          />
        ))}
      </div>
      
      {/* CSS Animations */}
      <style>{`
        @keyframes starMove {
          0% { transform: translateX(0) translateY(0); opacity: 0.6; }
          50% { opacity: 1; }
          100% { transform: translateX(-100px) translateY(-50px); opacity: 0; }
        }
        
        @keyframes nebulaDrift {
          0%, 100% { transform: translateX(0) translateY(0) scale(1); }
          25% { transform: translateX(50px) translateY(-30px) scale(1.1); }
          50% { transform: translateX(30px) translateY(40px) scale(0.9); }
          75% { transform: translateX(-40px) translateY(20px) scale(1.05); }
        }
        
        @keyframes shootingStar {
          0% { 
            transform: translateX(-100px) translateY(-100px) rotate(45deg) scale(0);
            opacity: 0;
          }
          10% { 
            transform: translateX(-50px) translateY(-50px) rotate(45deg) scale(1);
            opacity: 1;
          }
          90% { 
            transform: translateX(200px) translateY(200px) rotate(45deg) scale(1);
            opacity: 1;
          }
          100% { 
            transform: translateX(300px) translateY(300px) rotate(45deg) scale(0);
            opacity: 0;
          }
        }
        
        @keyframes galaxyPulse {
          0%, 100% { transform: scale(1); opacity: 0.4; }
          50% { transform: scale(1.5); opacity: 0.8; }
        }
        
        @keyframes dustFloat {
          0% { transform: translateY(0) translateX(0); opacity: 0.3; }
          25% { transform: translateY(-20px) translateX(10px); opacity: 0.6; }
          50% { transform: translateY(-10px) translateX(-5px); opacity: 0.4; }
          75% { transform: translateY(-30px) translateX(15px); opacity: 0.7; }
          100% { transform: translateY(-50px) translateX(20px); opacity: 0; }
        }
      `}</style>
    </div>
  )
}

export default CosmicGif
