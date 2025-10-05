import React, { useEffect, useRef, memo } from 'react';

interface Star {
  x: number;
  y: number;
  size: number;
  opacity: number;
  twinkleSpeed: number;
  twinklePhase: number;
}

const SimpleStarBackground: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Настройка canvas для полного экрана
    const resizeCanvas = () => {
      const dpr = window.devicePixelRatio || 1;
      const width = window.innerWidth;
      const height = window.innerHeight;
      
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = width + 'px';
      canvas.style.height = height + 'px';
      
      ctx.scale(dpr, dpr);
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    window.addEventListener('orientationchange', () => {
      setTimeout(resizeCanvas, 100);
    });

    // Создание звезд
    const stars: Star[] = [];
    const isMobile = window.innerWidth < 768;
    const isSmallMobile = window.innerWidth < 480;
    const starCount = isSmallMobile ? 80 : isMobile ? 120 : 150;

    for (let i = 0; i < starCount; i++) {
      stars.push({
        x: Math.random() * window.innerWidth,
        y: Math.random() * window.innerHeight,
        size: Math.random() * 1.5 + 0.5,
        opacity: Math.random() * 0.8 + 0.2,
        twinkleSpeed: Math.random() * 0.02 + 0.005,
        twinklePhase: Math.random() * Math.PI * 2
      });
    }

    // Анимация
    let animationId: number;
    let lastTime = 0;
    const targetFPS = isSmallMobile ? 20 : isMobile ? 25 : 30;
    const frameInterval = 1000 / targetFPS;
    
    const animate = (currentTime: number) => {
      if (currentTime - lastTime < frameInterval) {
        animationId = requestAnimationFrame(animate);
        return;
      }
      lastTime = currentTime;
      
      const canvasWidth = window.innerWidth;
      const canvasHeight = window.innerHeight;
      
      // Очистка с черным фоном
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, canvasWidth, canvasHeight);

      // Рисуем звёзды
      stars.forEach((star) => {
        const time = Date.now() * star.twinkleSpeed + star.twinklePhase;
        const twinkleOpacity = star.opacity + Math.sin(time) * 0.3;
        const twinkleSize = star.size + Math.sin(time * 2) * 0.3;

        // Основная звезда
        ctx.beginPath();
        ctx.arc(star.x, star.y, twinkleSize, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${Math.max(0, twinkleOpacity)})`;
        ctx.fill();

        // Легкое свечение для больших звезд
        if (star.size > 1) {
          ctx.beginPath();
          ctx.arc(star.x, star.y, twinkleSize * 1.5, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(255, 255, 255, ${Math.max(0, twinkleOpacity * 0.2)})`;
          ctx.fill();
        }
      });

      animationId = requestAnimationFrame(animate);
    };

    animate(0);

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      window.removeEventListener('orientationchange', resizeCanvas);
      cancelAnimationFrame(animationId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed top-0 left-0 w-screen h-screen z-0"
      style={{
        background: '#000000',
        pointerEvents: 'none',
        margin: 0,
        padding: 0,
        display: 'block'
      }}
    />
  );
};

export default memo(SimpleStarBackground);
