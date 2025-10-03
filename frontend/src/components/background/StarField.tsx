import React, { useCallback, useMemo } from 'react'
import Particles from 'react-tsparticles'
import { loadSlim } from 'tsparticles-slim'
import type { Container, Engine } from 'tsparticles-engine'
import { useTheme } from '../../contexts/ThemeContext'

const StarField: React.FC = React.memo(() => {
  const { theme } = useTheme()

  const particlesInit = useCallback(async (engine: Engine) => {
    await loadSlim(engine)
  }, [])

  const particlesLoaded = useCallback(async (container: Container | undefined) => {
    // Optional: Do something when particles are loaded
    if (process.env.NODE_ENV === 'development') {
      console.log('Particles loaded:', container)
    }
  }, [])

  const particlesOptions = useMemo(() => ({
    background: {
      color: {
        value: 'transparent',
      },
    },
    fpsLimit: 60, // Reduced FPS for better performance
    interactivity: {
      events: {
        onClick: {
          enable: false, // Disabled for performance
          mode: 'push',
        },
        onHover: {
          enable: false, // Disabled for performance
          mode: 'repulse',
        },
        resize: true,
      },
      modes: {
        push: {
          quantity: 0, // Disabled
        },
        repulse: {
          distance: 0, // Disabled
          duration: 0,
        },
      },
    },
    particles: {
      color: {
        value: theme === 'dark' ? '#ffffff' : '#1e293b',
      },
      links: {
        color: theme === 'dark' ? '#ffffff' : '#1e293b',
        distance: 150,
        enable: false, // Already disabled
        opacity: 0.1,
        width: 1,
      },
      collisions: {
        enable: false,
      },
      move: {
        direction: 'none' as const,
        enable: true,
        outModes: {
          default: 'out' as const, // Changed from bounce to out for performance
        },
        random: false, // Changed from true to false for performance
        speed: 0.2, // Reduced speed for performance
        straight: false,
      },
      number: {
        density: {
          enable: true,
          area: 1200, // Increased area for fewer particles
        },
        value: theme === 'dark' ? 100 : 50, // Reduced particle count for performance
      },
      opacity: {
        value: { min: 0.1, max: 0.8 }, // Reduced max opacity
        animation: {
          enable: true,
          speed: 0.5, // Slower animation
          minimumValue: 0.1,
          sync: false,
        },
      },
      shape: {
        type: 'circle',
      },
      size: {
        value: { min: 0.3, max: 1.5 }, // Smaller particle sizes
        animation: {
          enable: true,
          speed: 1, // Slower animation
          minimumValue: 0.3,
          sync: false,
        },
      },
    },
    detectRetina: true,
  }), [theme])

  // Memoize CSS stars to prevent re-rendering
  const cssStars = useMemo(() => (
    <>
      {Array.from({ length: 30 }).map((_, i) => (
        <div
          key={i}
          className={`
            absolute w-1 h-1 bg-white rounded-full animate-twinkle
            ${theme === 'light' ? 'opacity-30' : 'opacity-70'}
          `}
          style={{
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
            animationDelay: `${Math.random() * 3}s`,
            animationDuration: `${2 + Math.random() * 2}s`,
          }}
        />
      ))}
    </>
  ), [theme])

  // Memoize shooting stars to prevent re-rendering
  const shootingStars = useMemo(() => (
    <>
      {Array.from({ length: 2 }).map((_, i) => (
        <div
          key={`shooting-${i}`}
          className="absolute w-1 h-1 bg-gradient-to-r from-transparent via-white to-transparent rounded-full opacity-0"
          style={{
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 50}%`,
            animation: `shootingStar ${8 + Math.random() * 7}s linear infinite`,
            animationDelay: `${Math.random() * 15}s`,
          }}
        />
      ))}
    </>
  ), [])

  return (
    <div className="fixed inset-0 z-0 pointer-events-none">
      <Particles
        id="starfield"
        init={particlesInit}
        loaded={particlesLoaded}
        options={particlesOptions}
        className="w-full h-full"
      />
      
      {/* Additional CSS-based stars for better performance */}
      <div className="absolute inset-0 overflow-hidden">
        {cssStars}
      </div>
      
      {/* Shooting stars */}
      <div className="absolute inset-0 overflow-hidden">
        {shootingStars}
      </div>
      
      <style>{`
        @keyframes shootingStar {
          0% {
            opacity: 0;
            transform: translateX(-100px) translateY(0px);
          }
          10% {
            opacity: 1;
          }
          90% {
            opacity: 1;
          }
          100% {
            opacity: 0;
            transform: translateX(300px) translateY(100px);
          }
        }
        
        @keyframes twinkle {
          0%, 100% { opacity: 0.3; }
          50% { opacity: 1; }
        }
        
        .animate-twinkle {
          animation: twinkle 3s ease-in-out infinite;
        }
      `}</style>
    </div>
  )
})

export default StarField
