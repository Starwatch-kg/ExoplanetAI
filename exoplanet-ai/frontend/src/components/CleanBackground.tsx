import React from 'react'

const CleanBackground: React.FC = () => {
  return (
    <div className="fixed inset-0 pointer-events-none z-0">
      {/* Pure black background with subtle NASA blue accent */}
      <div className="absolute inset-0 bg-black">
        {/* Subtle blue glow in corners */}
        <div 
          className="absolute top-0 left-0 w-96 h-96 opacity-5"
          style={{
            background: 'radial-gradient(circle, #0B3D91 0%, transparent 70%)'
          }}
        />
        <div 
          className="absolute bottom-0 right-0 w-96 h-96 opacity-5"
          style={{
            background: 'radial-gradient(circle, #0B3D91 0%, transparent 70%)'
          }}
        />
      </div>
    </div>
  )
}

export default CleanBackground
