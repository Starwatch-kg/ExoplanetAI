/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class', // Enable class-based dark mode
  theme: {
    extend: {
      colors: {
        // NASA Space Theme Colors
        nasa: {
          blue: '#4A5568',
          red: '#FC3D21',
          white: '#FFFFFF',
        },
        space: {
          black: '#000000',
          deep: '#0a0a0a',
          nebula: '#2D3748',
          cosmic: '#2D3748',
          stellar: '#4A5568',
          galaxy: '#718096',
        },
        neon: {
          cyan: '#00FFFF',
          pink: '#FF006E',
          orange: '#FF8500',
          green: '#39FF14',
        },
        starlight: '#E6E6FA',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'float': 'float 6s ease-in-out infinite',
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'twinkle': 'twinkle 3s infinite',
        'orbit': 'orbit 20s linear infinite',
        'spin-slow': 'spin 3s linear infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-20px)' },
        },
        glow: {
          '0%': { boxShadow: '0 0 5px #9CA3AF, 0 0 10px #9CA3AF, 0 0 15px #9CA3AF' },
          '100%': { boxShadow: '0 0 10px #9CA3AF, 0 0 20px #9CA3AF, 0 0 30px #9CA3AF' },
        },
        twinkle: {
          '0%': { opacity: '0.3' },
          '50%': { opacity: '1' },
          '100%': { opacity: '0.3' },
        },
        orbit: {
          '0%': { transform: 'rotate(0deg) translateX(100px) rotate(0deg)' },
          '100%': { transform: 'rotate(360deg) translateX(100px) rotate(-360deg)' },
        },
      },
      backdropBlur: {
        xs: '2px',
        '3xl': '64px',
      },
      boxShadow: {
        'neon-blue': '0 0 20px rgba(156, 163, 175, 0.5), 0 0 40px rgba(156, 163, 175, 0.3), 0 0 60px rgba(156, 163, 175, 0.1)',
        'neon-purple': '0 0 20px rgba(156, 163, 175, 0.5), 0 0 40px rgba(156, 163, 175, 0.3), 0 0 60px rgba(156, 163, 175, 0.1)',
        'neon-cyan': '0 0 20px rgba(251, 211, 141, 0.5), 0 0 40px rgba(251, 211, 141, 0.3)',
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
      },
    },
  },
  plugins: [],
}
