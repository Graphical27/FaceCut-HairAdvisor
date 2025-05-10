import { useState, useRef, useEffect } from 'react'
import Webcam from 'react-webcam'
import axios from 'axios'
import './styles.css'


export default function App() {
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('camera')
  const [appState, setAppState] = useState('home') // 'home', 'input', 'results'
  const [analyzerVisible, setAnalyzerVisible] = useState(false)
  const webcamRef = useRef(null)
  const fileInputRef = useRef(null)
  
  // References for animations
  const homeRef = useRef(null)
  const analyzerRef = useRef(null)
  
  useEffect(() => {
    // Add animation class when analyzer becomes visible
      if (analyzerVisible && analyzerRef.current) {
      analyzerRef.current.classList.add('analyzer-appear')
    }
  }, [analyzerVisible])
  
  const startAnalyzer = () => {
    if (homeRef.current) {
      homeRef.current.classList.add('fade-out')
      setTimeout(() => {
        setAnalyzerVisible(true)
        setAppState('input')
      }, 500)
    }
  }
  
  const sendFile = async (file) => {
    setLoading(true)
    const formData = new FormData()
    formData.append('file', file)
    try {
      const res = await axios.post('http://localhost:8000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setResult(res.data)
      setAppState('results')
      setLoading(false)
    } catch (err) {
      console.error(err)
      setLoading(false)
    }
  }

  const onUpload = (e) => {
    const file = e.target.files[0]
    if (!file) return
    const url = URL.createObjectURL(file)
    setPreview(url)
    sendFile(file)
  }


  const capture = async () => {
    const imageSrc = webcamRef.current.getScreenshot()
    if (!imageSrc) return
    setPreview(imageSrc)
    const response = await fetch(imageSrc)
    const blob = await response.blob()
    const file = new File([blob], 'selfie.jpg', { type: blob.type })
    sendFile(file)
  }

  const triggerFileInput = () => {
    fileInputRef.current.click()
  }
  
  const resetApp = () => {
    setAppState('home')
    setAnalyzerVisible(false)
    setPreview(null)
    setResult(null)
    if (homeRef.current) {
      homeRef.current.classList.remove('fade-out')
    }
  }

  return (
    <div className="app-wrapper">
      {appState === 'home' && (
        <div className="home-container" ref={homeRef}>
          <header className="home-header">
            <h1 className="title-animation">Face Shape Analyzer</h1>
            <p className="subtitle-animation">Discover your face shape and suitable style recommendations</p>
          </header>
          
          <div className="face-shapes-container">
            <h2 className="section-title">Understanding Face Shapes</h2>
            
            <div className="face-shapes-grid">
              <div className="face-shape-card">
                <div className="shape-icon oval">◯</div>
                <h3>Oval</h3>
                <p>Balanced proportions with a slightly narrower forehead and jaw.</p>
              </div>
              
              <div className="face-shape-card">
                <div className="shape-icon round">⬤</div>
                <h3>Round</h3>
                <p>Similar width and length with soft curves and fuller cheeks.</p>
              </div>
              
              <div className="face-shape-card">
                <div className="shape-icon square">◼</div>
                <h3>Square</h3>
                <p>Strong jawline with a forehead that's similar in width.</p>
              </div>
              
              <div className="face-shape-card">
                <div className="shape-icon heart">♥</div>
                <h3>Heart</h3>
                <p>Wider forehead that narrows down to a pointed chin.</p>
              </div>
              
              <div className="face-shape-card">
                <div className="shape-icon diamond">◆</div>
                <h3>Diamond</h3>
                <p>Narrow forehead and jawline with wider cheekbones.</p>
              </div>
              
              <div className="face-shape-card">
                <div className="shape-icon rectangle">▭</div>
                <h3>Rectangle</h3>
                <p>Longer face with a forehead, cheeks, and jawline of similar width.</p>
              </div>
            </div>
            
            <div className="analyzer-cta">
              <button className="primary-button pulse-animation" onClick={startAnalyzer}>
                Check Your Face Shape
              </button>
            </div>
          </div>
          
          <div className="benefits-section">
            <h2 className="section-title">Why Know Your Face Shape?</h2>
            <div className="benefits-grid">
              <div className="benefit-card">
                <div className="benefit-icon">💇</div>
                <h3>Perfect Hairstyle</h3>
                <p>Find hairstyles that complement your natural features</p>
              </div>
              
              <div className="benefit-card">
                <div className="benefit-icon">👓</div>
                <h3>Ideal Eyewear</h3>
                <p>Choose glasses and sunglasses that balance your proportions</p>
              </div>
              
              <div className="benefit-card">
                <div className="benefit-icon">💍</div>
                <h3>Flattering Accessories</h3>
                <p>Select earrings and accessories that enhance your look</p>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {analyzerVisible && (
        <div className="analyzer-container" ref={analyzerRef}>
          <button className="back-button" onClick={resetApp}>← Back to Home</button>
          
          <div className="analyzer-card">
            <div className="analyzer-header">
              <h2>Analyze Your Face Shape</h2>
              <p className="analyzer-desc">Take or upload a front-facing photo for accurate results</p>
            </div>
            
            <div className="tab-navigation">
              <button 
                className={`tab-button ${activeTab === 'camera' ? 'active' : ''}`} 
                onClick={() => setActiveTab('camera')}
              >
                Use Camera
              </button>
              <button 
                className={`tab-button ${activeTab === 'upload' ? 'active' : ''}`} 
                onClick={() => setActiveTab('upload')}
              >
                Upload Photo
              </button>
            </div>
            
            <div className="analyzer-content">
              {activeTab === 'camera' && !loading && !preview && (
                <div className="camera-container">
                  <div className="webcam-wrapper">
                    <Webcam
                      audio={false}
                      ref={webcamRef}
                      screenshotFormat="image/jpeg"
                      videoConstraints={{ facingMode: 'user' }}
                      className="webcam"
                    />
                    <div className="face-outline-overlay">
                      <div className="face-outline"></div>
                    </div>
                  </div>
                  <p className="camera-tip">Position your face within the outline and look straight ahead</p>
                  <button className="capture-button" onClick={capture}>
                    <span className="camera-icon">📸</span>
                    Take Photo
                  </button>
                </div>
              )}
              
              {activeTab === 'upload' && !loading && !preview && (
                <div className="upload-container">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={onUpload}
                    ref={fileInputRef}
                    className="hidden-input"
                  />
                  <div className="upload-area" onClick={triggerFileInput}>
                    <div className="upload-icon">
                      <svg width="60" height="60" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 5V19M12 5L7 10M12 5L17 10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        <path d="M20 21H4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    </div>
                    <p className="upload-text">Click to select an image<br /><span className="upload-subtext">or drag and drop</span></p>
                  </div>
                </div>
              )}
              
              {loading && (
                <div className="loading-container">
                  <div className="loader"></div>
                  <p className="loading-text">Analyzing your face shape...</p>
                </div>
              )}
              
              {preview && !loading && appState === 'results' && (
                <div className="results-container">
                  <div className="results-grid">
                    <div className="result-preview">
                      <img src={preview} alt="Your face" className="result-image" />
                      <div className="image-overlay"></div>
                    </div>
                    
                    <div className="result-details">
                      <div className="result-item">
                        <h3>Your Face Shape</h3>
                        <p className="result-value">{result?.shape || "Unknown"}</p>
                      </div>
                      
                      <div className="result-item">
                        <h3>Gender Analysis</h3>
                        <p className="result-value">{result?.gender || "Unknown"}</p>
                      </div>
                      
                      <div className="recommended-styles">
                        <h3>Recommendations</h3>
                        <div className="recommendations">
                          <div className="rec-item">
                            <span className="rec-icon">💇</span>
                            <span className="rec-text">
                              {result?.shape === "Oval" && "Most hairstyles suit this versatile shape"}
                              {result?.shape === "Round" && "Layered cuts add definition"}
                              {result?.shape === "Square" && "Soft layers to balance strong jawline"}
                              {result?.shape === "Heart" && "Side-swept bangs complement wider forehead"}
                              {result?.shape === "Diamond" && "Hairstyles with volume around the jawline"}
                              {result?.shape === "Rectangle" && "Cuts with volume on the sides"}
                              {!result?.shape && "Personalized hairstyle suggestions"}
                            </span>
                          </div>
                          <div className="rec-item">
                            <span className="rec-icon">👓</span>
                            <span className="rec-text">
                              {result?.shape === "Oval" && "Most styles work well"}
                              {result?.shape === "Round" && "Angular frames add definition"}
                              {result?.shape === "Square" && "Round or oval frames soften features"}
                              {result?.shape === "Heart" && "Frames wider at the bottom balance proportions"}
                              {result?.shape === "Diamond" && "Oval or rimless styles complement angles"}
                              {result?.shape === "Rectangle" && "Oversized frames shorten face appearance"}
                              {!result?.shape && "Ideal eyewear for your face shape"}
                            </span>
                          </div>
                        </div>
                      </div>
                      
                      <button className="try-again-button" onClick={resetApp}>
                        Try Again
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
