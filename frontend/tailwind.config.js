/** @type {import('tailwindcss').Config} */
<<<<<<< HEAD
<<<<<<< HEAD
module.exports = {
=======
export default {
>>>>>>> ef5c656 (Версия 1.5.1)
=======
export default {
>>>>>>> 975c3a7 (Версия 1.5.1)
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
<<<<<<< HEAD
<<<<<<< HEAD
        'space': {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
          950: '#082f49',
        },
        'cosmic': {
          50: '#faf5ff',
          100: '#f3e8ff',
          200: '#e9d5ff',
          300: '#d8b4fe',
          400: '#c084fc',
          500: '#a855f7',
          600: '#9333ea',
          700: '#7c3aed',
          800: '#6b21a8',
          900: '#581c87',
          950: '#3b0764',
        }
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float': 'float 6s ease-in-out infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'orbit': 'orbit 20s linear infinite',
        'twinkle': 'twinkle 2s ease-in-out infinite alternate',
=======
=======
>>>>>>> 975c3a7 (Версия 1.5.1)
        primary: {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
        },
        space: {
          50: '#0f0f23',
          100: '#1a1a3a',
          200: '#252550',
          300: '#303066',
          400: '#3b3b7d',
          500: '#464693',
          600: '#5151aa',
          700: '#5c5cc0',
          800: '#6767d6',
          900: '#7272ed',
        },
        cosmic: {
          50: '#fdf2f8',
          100: '#fce7f3',
          200: '#fbcfe8',
          300: '#f9a8d4',
          400: '#f472b6',
          500: '#ec4899',
          600: '#db2777',
          700: '#be185d',
          800: '#9d174d',
          900: '#831843',
        }
      },
      animation: {
        'float': 'float 6s ease-in-out infinite',
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.5s ease-out',
<<<<<<< HEAD
>>>>>>> ef5c656 (Версия 1.5.1)
=======
>>>>>>> 975c3a7 (Версия 1.5.1)
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
<<<<<<< HEAD
<<<<<<< HEAD
          '50%': { transform: 'translateY(-20px)' },
        },
        glow: {
          '0%': { boxShadow: '0 0 5px #0ea5e9, 0 0 10px #0ea5e9, 0 0 15px #0ea5e9' },
          '100%': { boxShadow: '0 0 10px #0ea5e9, 0 0 20px #0ea5e9, 0 0 30px #0ea5e9' },
        },
        orbit: {
          '0%': { transform: 'rotate(0deg) translateX(100px) rotate(0deg)' },
          '100%': { transform: 'rotate(360deg) translateX(100px) rotate(-360deg)' },
        },
        twinkle: {
          '0%': { opacity: '0.3' },
          '100%': { opacity: '1' },
        }
=======
=======
>>>>>>> 975c3a7 (Версия 1.5.1)
          '50%': { transform: 'translateY(-10px)' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
      backgroundImage: {
        'space-gradient': 'linear-gradient(135deg, #0f0f23 0%, #1a1a3a 50%, #252550 100%)',
        'cosmic-gradient': 'linear-gradient(135deg, #831843 0%, #be185d 50%, #ec4899 100%)',
<<<<<<< HEAD
>>>>>>> ef5c656 (Версия 1.5.1)
=======
>>>>>>> 975c3a7 (Версия 1.5.1)
      }
    },
  },
  plugins: [],
}
