import React, { useEffect, useRef, memo } from 'react';

interface Star {
  x: number;
  y: number;
  size: number;
  opacity: number;
  twinkleSpeed: number;
  twinklePhase: number;
}

interface ShootingStar {
  x: number;
  y: number;
  speed: number;
  life: number;
  maxLife: number;
  trail: Array<{x: number, y: number, opacity: number}>;
}

interface Nebula {
  x: number;
  y: number;
  size: number;
  opacity: number;
  color: string;
  pulseSpeed: number;
  pulsePhase: number;
}

const StarBackground: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Настройка canvas с оптимизацией для полного экрана
    const resizeCanvas = () => {
      const dpr = window.devicePixelRatio || 1;
      
      // Используем размеры окна вместо getBoundingClientRect для полного покрытия
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
      // Небольшая задержка для корректного получения новых размеров после поворота
      setTimeout(resizeCanvas, 100);
    });

    // Создание объектов
    const stars: Star[] = [];
    const shootingStars: ShootingStar[] = [];
    const nebulas: Nebula[] = [];

    // Генерация звезд (адаптивное количество для производительности)
    const isMobile = window.innerWidth < 768;
    const isSmallMobile = window.innerWidth < 480;
    const starCount = isSmallMobile ? 100 : isMobile ? 150 : 200;
    for (let i = 0; i < starCount; i++) {
      stars.push({
        x: Math.random() * window.innerWidth,
        y: Math.random() * window.innerHeight,
        size: Math.random() * 2 + 0.5,
        opacity: Math.random() * 0.8 + 0.2,
        twinkleSpeed: Math.random() * 0.02 + 0.005,
        twinklePhase: Math.random() * Math.PI * 2
      });
    }

    // Генерация туманностей
    const nebulaColors = [
      'rgba(147, 51, 234, 0.3)', // purple
      'rgba(59, 130, 246, 0.3)', // blue
      'rgba(16, 185, 129, 0.3)', // green
      'rgba(245, 101, 101, 0.3)', // red
      'rgba(251, 146, 60, 0.3)'  // orange
    ];

    // Адаптивное количество туманностей для производительности
    const nebulaCount = isSmallMobile ? 2 : isMobile ? 3 : 5;
    for (let i = 0; i < nebulaCount; i++) {
      nebulas.push({
        x: Math.random() * window.innerWidth,
        y: Math.random() * window.innerHeight,
        size: Math.random() * 150 + 80,
        opacity: Math.random() * 0.3 + 0.1,
        color: nebulaColors[Math.floor(Math.random() * nebulaColors.length)],
        pulseSpeed: Math.random() * 0.008 + 0.003,
        pulsePhase: Math.random() * Math.PI * 2
      });
    }

    // Функция создания падающей звезды
    const createShootingStar = () => {
      if (Math.random() < 0.015) { // 1.5% шанс каждые 50мс
        shootingStars.push({
          x: -20,
          y: Math.random() * window.innerHeight * 0.4,
          speed: Math.random() * 4 + 2,
          life: 0,
          maxLife: 120,
          trail: []
        });
      }
    };

    // Анимация с адаптивной оптимизацией FPS
    let animationId: number;
    let lastTime = 0;
    // Адаптивный FPS: меньше на мобильных для экономии батареи
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
      
      ctx.clearRect(0, 0, canvasWidth, canvasHeight);

      // Фон космоса
      const gradient = ctx.createRadialGradient(
        canvasWidth / 2, canvasHeight / 2, 0,
        canvasWidth / 2, canvasHeight / 2, Math.max(canvasWidth, canvasHeight) / 1.5
      );
      gradient.addColorStop(0, '#0a0b1e');
      gradient.addColorStop(0.5, '#1a0b2e');
      gradient.addColorStop(1, '#000');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, canvasWidth, canvasHeight);

      // Рисуем туманности
      nebulas.forEach(nebula => {
        const time = Date.now() * nebula.pulseSpeed + nebula.pulsePhase;
        const pulseOpacity = nebula.opacity + Math.sin(time) * 0.2;

        const nebulaGradient = ctx.createRadialGradient(
          nebula.x, nebula.y, 0,
          nebula.x, nebula.y, nebula.size
        );
        nebulaGradient.addColorStop(0, nebula.color.replace('0.3', pulseOpacity.toString()));
        nebulaGradient.addColorStop(1, 'transparent');

        ctx.beginPath();
        ctx.arc(nebula.x, nebula.y, nebula.size, 0, Math.PI * 2);
        ctx.fillStyle = nebulaGradient;
        ctx.fill();
      });

      // Рисуем звёзды
      stars.forEach((star) => {
        const time = Date.now() * star.twinkleSpeed + star.twinklePhase;
        const twinkleOpacity = star.opacity + Math.sin(time) * 0.4;
        const twinkleSize = star.size + Math.sin(time * 2) * 0.5;

        // Основная звезда
        ctx.beginPath();
        ctx.arc(star.x, star.y, twinkleSize, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${twinkleOpacity})`;
        ctx.fill();

        // Свечение
        ctx.beginPath();
        ctx.arc(star.x, star.y, twinkleSize * 2, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(135, 206, 250, ${twinkleOpacity * 0.3})`;
        ctx.fill();

        // Блик
        if (twinkleOpacity > 0.8) {
          ctx.beginPath();
          ctx.arc(star.x - 1, star.y - 1, twinkleSize * 0.5, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(255, 255, 255, ${twinkleOpacity * 0.8})`;
          ctx.fill();
        }
      });

      // Рисуем падающие звёзды
      shootingStars.forEach((shootingStar, index) => {
        const alpha = shootingStar.life / shootingStar.maxLife;

        // Обновляем след
        shootingStar.trail.unshift({
          x: shootingStar.x,
          y: shootingStar.y,
          opacity: alpha
        });

        if (shootingStar.trail.length > 15) {
          shootingStar.trail.pop();
        }

        // Рисуем след
        shootingStar.trail.forEach((trailPoint, trailIndex) => {
          const trailAlpha = trailPoint.opacity * (1 - trailIndex / shootingStar.trail.length);
          ctx.beginPath();
          ctx.arc(trailPoint.x, trailPoint.y, 1, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(255, 255, 255, ${trailAlpha})`;
          ctx.fill();
        });

        // Рисуем саму звезду
        ctx.beginPath();
        ctx.moveTo(shootingStar.x, shootingStar.y);
        ctx.lineTo(shootingStar.x + 25, shootingStar.y + 25);
        ctx.strokeStyle = `rgba(255, 255, 255, ${alpha})`;
        ctx.lineWidth = 3;
        ctx.stroke();

        // Обновляем позицию
        shootingStar.x += shootingStar.speed;
        shootingStar.y += shootingStar.speed;
        shootingStar.life++;

        // Удаляем если вышла за пределы
        if (shootingStar.life >= shootingStar.maxLife) {
          shootingStars.splice(index, 1);
        }
      });

      createShootingStar();
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
        background: 'radial-gradient(ellipse at center, #0a0b1e 0%, #1a0b2e 50%, #000 100%)',
        pointerEvents: 'none', // Позволяет кликать через фон
        margin: 0,
        padding: 0,
        display: 'block'
      }}
    />
  );
};

export default memo(StarBackground);
