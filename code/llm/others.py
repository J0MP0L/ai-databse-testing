loading_code = """
<style>
@keyframes glitch {
  0%, 100% { clip-path: inset(0 0 0 0); }
  20% { clip-path: inset(10% 0 85% 0); }
  40% { clip-path: inset(50% 0 30% 0); }
  60% { clip-path: inset(30% 0 50% 0); }
  80% { clip-path: inset(85% 0 10% 0); }
}

@keyframes scan {
  0% { transform: translateY(-100%); }
  100% { transform: translateY(100%); }
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.loading-container {
  position: relative;
  padding: 15px 20px;  
  background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
  border: 1px solid #00ffff;  
  border-radius: 8px;  
  box-shadow: 0 0 15px rgba(0, 255, 255, 0.3), inset 0 0 15px rgba(0, 255, 255, 0.1); 
  overflow: hidden;
  max-width: 400px;  
  margin: 0 auto; 
}

.loading-container::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 1px; 
  background: linear-gradient(90deg, transparent, #00ffff, transparent);
  animation: scan 2s linear infinite;
}

.spinner-container {
  display: flex;
  align-items: center;
  gap: 12px; 
}

.hexagon-spinner {
  position: relative;
  width: 30px; 
  height: 30px;  
  flex-shrink: 0;  
}

.hexagon {
  position: absolute;
  width: 30px; 
  height: 30px;  
  border: 2px solid transparent;  
  border-top-color: #00ffff;
  border-bottom-color: #ff00ff;
  clip-path: polygon(30% 0%, 70% 0%, 100% 50%, 70% 100%, 30% 100%, 0% 50%);
  animation: rotate 2s linear infinite;
}

.hexagon:nth-child(2) {
  animation-delay: -0.5s;
  opacity: 0.6;
}

.hexagon:nth-child(3) {
  animation-delay: -1s;
  opacity: 0.3;
}

@keyframes rotate {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.loading-text {
  font-family: 'Courier New', monospace;
  font-size: 14px;  
  font-weight: bold;
  color: #00ffff;
  text-shadow: 0 0 5px rgba(0, 255, 255, 0.8), 0 0 10px rgba(0, 255, 255, 0.6);  
  animation: glitch 3s infinite, pulse 2s infinite;
}

.loading-dots {
  display: inline-block;
}

.loading-dots::after {
  content: '';
  animation: dots 1.5s steps(4, end) infinite;
}

@keyframes dots {
  0%, 20% { content: ''; }
  40% { content: '.'; }
  60% { content: '..'; }
  80%, 100% { content: '...'; }
}

.progress-bar {
  width: 100%;
  height: 2px;  
  background: rgba(0, 255, 255, 0.1);
  border-radius: 1px;
  margin-top: 10px; 
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #00ffff, #ff00ff, #00ffff);
  background-size: 200% 100%;
  animation: progress 2s linear infinite;
}

@keyframes progress {
  0% { background-position: 0% 0%; }
  100% { background-position: 200% 0%; }
}
</style>

<div class="loading-container">
  <div class="spinner-container">
    <div class="hexagon-spinner">
      <div class="hexagon"></div>
      <div class="hexagon"></div>
      <div class="hexagon"></div>
    </div>
    <div class="loading-text">
      Generating Graph<span class="loading-dots"></span>
    </div>
  </div>
  <div class="progress-bar">
    <div class="progress-fill"></div>
  </div>
</div>
"""