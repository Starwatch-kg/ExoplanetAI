import React, { useEffect, useRef } from 'react';

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

    // Настройка canvas
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Создание объектов
    const stars: Star[] = [];
    const shootingStars: ShootingStar[] = [];
    const nebulas: Nebula[] = [];

    // Генерация звезд
    for (let i = 0; i < 300; i++) {
      stars.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        size: Math.random() * 3 + 0.5,
        opacity: Math.random() * 0.8 + 0.2,
        twinkleSpeed: Math.random() * 0.03 + 0.005,
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

    for (let i = 0; i < 8; i++) {
      nebulas.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        size: Math.random() * 200 + 100,
        opacity: Math.random() * 0.4 + 0.1,
        color: nebulaColors[Math.floor(Math.random() * nebulaColors.length)],
        pulseSpeed: Math.random() * 0.01 + 0.005,
        pulsePhase: Math.random() * Math.PI * 2
      });
    }

    // Функция создания падающей звезды
    const createShootingStar = () => {
      if (Math.random() < 0.015) { // 1.5% шанс каждые 50мс
        shootingStars.push({
          x: -20,
          y: Math.random() * canvas.height * 0.4,
          speed: Math.random() * 4 + 2,
          life: 0,
          maxLife: 120,
          trail: []
        });
      }
    };

    // Анимация
    let animationId: number;
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Фон космоса
      const gradient = ctx.createRadialGradient(
        canvas.width / 2, canvas.height / 2, 0,
        canvas.width / 2, canvas.height / 2, Math.max(canvas.width, canvas.height) / 1.5
      );
      gradient.addColorStop(0, '#0a0b1e');
      gradient.addColorStop(0.5, '#1a0b2e');
      gradient.addColorStop(1, '#000');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

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

    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      cancelAnimationFrame(animationId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed top-0 left-0 w-full h-full -z-10"
      style={{
        background: 'radial-gradient(ellipse at center, #0a0b1e 0%, #1a0b2e 50%, #000 100%)'
      }}
    />
  );
};

export default StarBackground;
