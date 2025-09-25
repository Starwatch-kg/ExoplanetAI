import React, { useEffect, useRef } from 'react'

interface Star {
  x: number
  y: number
  size: 'small' | 'medium' | 'large'
  delay: number
}

const StarField: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null)
  const starsRef = useRef<Star[]>([])

  useEffect(() => {
    const generateStars = () => {
      const stars: Star[] = []
      const starCount = 150

      for (let i = 0; i < starCount; i++) {
        const size = Math.random() < 0.7 ? 'small' : Math.random() < 0.9 ? 'medium' : 'large'
        stars.push({
          x: Math.random() * 100,
          y: Math.random() * 100,
          size,
          delay: Math.random() * 4
        })
      }

      starsRef.current = stars
    }

    generateStars()
  }, [])

  return (
    <div ref={containerRef} className="stars-container">
      {starsRef.current.map((star, index) => (
        <div
          key={index}
          className={`star star-${star.size}`}
          style={{
            left: `${star.x}%`,
            top: `${star.y}%`,
            animationDelay: `${star.delay}s`
          }}
        />
      ))}
    </div>
  )
}

export default StarField
