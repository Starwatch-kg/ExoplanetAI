import React, { useCallback, useMemo } from 'react'
import Particles from 'react-tsparticles'
import { loadSlim } from 'tsparticles-slim'
import type { Container, Engine } from 'tsparticles-engine'
import { useTheme } from '../../contexts/ThemeContext'

const StarField: React.FC = () => {
  const { theme } = useTheme()

  const particlesInit = useCallback(async (engine: Engine) => {
    await loadSlim(engine)
  }, [])

  const particlesLoaded = useCallback(async (container: Container | undefined) => {
    // Optional: Do something when particles are loaded
    console.log('Particles loaded:', container)
  }, [])

  const particlesOptions = useMemo(() => ({
    background: {
      color: {
        value: 'transparent',
      },
    },
    fpsLimit: 120,
    interactivity: {
      events: {
        onClick: {
          enable: true,
          mode: 'push',
        },
        onHover: {
          enable: true,
          mode: 'repulse',
        },
        resize: true,
      },
      modes: {
        push: {
          quantity: 4,
        },
        repulse: {
          distance: 200,
          duration: 0.4,
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
        enable: false, // Disable links for cleaner star field
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
          default: 'bounce' as const,
        },
        random: true,
        speed: 0.5,
        straight: false,
      },
      number: {
        density: {
          enable: true,
          area: 800,
        },
        value: theme === 'dark' ? 150 : 80, // More stars in dark mode
      },
      opacity: {
        value: { min: 0.1, max: 1 },
        animation: {
          enable: true,
          speed: 1,
          minimumValue: 0.1,
          sync: false,
        },
      },
      shape: {
        type: 'circle',
      },
      size: {
        value: { min: 0.5, max: 2 },
        animation: {
          enable: true,
          speed: 2,
          minimumValue: 0.5,
          sync: false,
        },
      },
    },
    detectRetina: true,
  }), [theme])

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
        {Array.from({ length: 50 }).map((_, i) => (
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
      </div>
      
      {/* Shooting stars */}
      <div className="absolute inset-0 overflow-hidden">
        {Array.from({ length: 3 }).map((_, i) => (
          <div
            key={`shooting-${i}`}
            className="absolute w-1 h-1 bg-gradient-to-r from-transparent via-white to-transparent rounded-full opacity-0"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 50}%`,
              animation: `shootingStar ${5 + Math.random() * 5}s linear infinite`,
              animationDelay: `${Math.random() * 10}s`,
            }}
          />
        ))}
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
      `}</style>
    </div>
  )
}

export default StarField
