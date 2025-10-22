import { useCallback, useMemo, useRef, useEffect, useState } from 'react';

/**
 * Hook для оптимизации производительности компонентов
 */
export const usePerformanceOptimization = () => {
  const renderCountRef = useRef(0);
  const lastRenderTimeRef = useRef(Date.now());

  // Debounce функция для оптимизации частых обновлений
  const useDebounce = useCallback((callback: Function, delay: number) => {
    const timeoutRef = useRef<NodeJS.Timeout>();

    return useCallback((...args: any[]) => {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = setTimeout(() => callback(...args), delay);
    }, [callback, delay]);
  }, []);

  // Throttle функция для ограничения частоты вызовов
  const useThrottle = useCallback((callback: Function, limit: number) => {
    const inThrottle = useRef(false);

    return useCallback((...args: any[]) => {
      if (!inThrottle.current) {
        callback(...args);
        inThrottle.current = true;
        setTimeout(() => inThrottle.current = false, limit);
      }
    }, [callback, limit]);
  }, []);

  // Мемоизация тяжелых вычислений
  const useMemoizedCalculation = useCallback((
    calculation: () => any,
    dependencies: any[]
  ) => {
    return useMemo(calculation, dependencies);
  }, []);

  // Отслеживание производительности рендеринга
  useEffect(() => {
    renderCountRef.current += 1;
    const now = Date.now();
    const timeSinceLastRender = now - lastRenderTimeRef.current;
    lastRenderTimeRef.current = now;

    // Предупреждение о частых рендерах
    if (timeSinceLastRender < 16 && renderCountRef.current > 10) {
      console.warn('Частые рендеры компонента, рассмотрите оптимизацию');
    }
  });

  return {
    useDebounce,
    useThrottle,
    useMemoizedCalculation,
    renderCount: renderCountRef.current
  };
};

/**
 * Hook для оптимизации работы с изображениями
 */
export const useImageOptimization = () => {
  const lazyLoadImage = useCallback((src: string, placeholder?: string) => {
    const [imageSrc, setImageSrc] = useState(placeholder || '');
    const [isLoaded, setIsLoaded] = useState(false);
    const imgRef = useRef<HTMLImageElement>();

    useEffect(() => {
      const img = new Image();
      img.onload = () => {
        setImageSrc(src);
        setIsLoaded(true);
      };
      img.src = src;
      imgRef.current = img;

      return () => {
        if (imgRef.current) {
          imgRef.current.onload = null;
        }
      };
    }, [src]);

    return { imageSrc, isLoaded };
  }, []);

  return { lazyLoadImage };
};

/**
 * Hook для оптимизации анимаций
 */
export const useAnimationOptimization = () => {
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setPrefersReducedMotion(mediaQuery.matches);

    const handler = (e: MediaQueryListEvent) => {
      setPrefersReducedMotion(e.matches);
    };

    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);

  const getAnimationProps = useCallback((animationProps: any) => {
    if (prefersReducedMotion) {
      return {
        ...animationProps,
        animate: false,
        transition: { duration: 0 }
      };
    }
    return animationProps;
  }, [prefersReducedMotion]);

  return { prefersReducedMotion, getAnimationProps };
};
