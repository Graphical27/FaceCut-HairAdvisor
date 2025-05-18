import { useState, useRef, useEffect } from 'react'
import Webcam from 'react-webcam'
import axios from 'axios'
import './styles.css'

export default function App() {
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('camera')
  const [appState, setAppState] = useState('home')
  const [analyzerVisible, setAnalyzerVisible] = useState(false)
  const [darkMode, setDarkMode] = useState(false)
  const webcamRef = useRef(null)
  const fileInputRef = useRef(null)

  const homeRef = useRef(null)
  const analyzerRef = useRef(null)

  useEffect(() => {
    if (analyzerVisible && analyzerRef.current) {
      analyzerRef.current.classList.add('analyzer-appear')
    }
  }, [analyzerVisible])

  // Apply dark mode class to body
  useEffect(() => {
    if (darkMode) {
      document.body.classList.add('dark-mode')
    } else {
      document.body.classList.remove('dark-mode')
    }
  }, [darkMode])

  const toggleDarkMode = () => {
    setDarkMode(!darkMode)
  }

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

  const getImagePaths = () => {
    if (!result?.gender || !result?.shape) return []
    const gender = result.gender.toLowerCase()
    const shape = result.shape.charAt(0).toUpperCase() + result.shape.slice(1).toLowerCase()
    return Array.from({ length: 5 }, (_, i) =>
      `/images/${gender}/${shape}/Style ${i + 1}.jpg`
    )
  }

  // Footer component with developer information
  const Footer = () => (
    <footer className="footer">
      <div className="footer-container">
        <h3 className="footer-title">Developers</h3>
        <div className="developers-list">
          <div className="developer-item">
            <div className="developer-name">Sourabh Singh</div>
            <a 
              href="https://github.com/Graphical27" 
              target="_blank" 
              rel="noopener noreferrer" 
              className="developer-github"
            >
              github.com/Graphical27
            </a>
          </div>
          <div className="developer-item">
            <div className="developer-name">Paras Mheta</div>
            <a 
              href="https://github.com/Paras-Mehta007" 
              target="_blank" 
              rel="noopener noreferrer" 
              className="developer-github"
            >
              github.com/Paras-Mehta007
            </a>
          </div>
          <div className="developer-item">
            <div className="developer-name">Gaurav Singh</div>
            <a 
              href="https://github.com/gauravsinghshah" 
              target="_blank" 
              rel="noopener noreferrer" 
              className="developer-github"
            >
              github.com/gauravsinghshah
            </a>
          </div>
        </div>
        <div className="footer-copyright">
          © {new Date().getFullYear()} Perfect Cut - All Rights Reserved
        </div>
      </div>
    </footer>
  )

  return (
    <div className={`app-container ${darkMode ? 'dark-mode' : ''}`}>
      <div className="theme-toggle">
        <button 
          className="theme-toggle-button" 
          onClick={toggleDarkMode}
          title={darkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
        >
          {darkMode ? '☀️' : '🌙'}
        </button>
      </div>

      {appState === 'home' && (
        <div className="home-screen" ref={homeRef}>
          <div className="home-content">
            <h1 className="app-title">Perfect Cut</h1>
            <p className="app-subtitle">Find your ideal hairstyle based on your face shape</p>
            <button className="start-button" onClick={startAnalyzer}>
              <span className="button-icon">✓</span>
              <span className="button-text">Find My Perfect Hairstyle</span>
            </button>
          </div>
          
          {/* Footer only on main/home page */}
          <Footer />
        </div>
      )}

      {analyzerVisible && (
        <div className="analyzer" ref={analyzerRef}>
          <button className="back-button" onClick={resetApp}>
            <span className="back-icon">←</span>
            <span className="back-text">Back</span>
          </button>
          
          <div className="analyzer-panel">
            <h2 className="panel-title">Face Shape Analyzer</h2>
            
            <div className="tabs">
              <button 
                className={`tab ${activeTab === 'camera' ? 'active-tab' : ''}`} 
                onClick={() => setActiveTab('camera')}
              >
                <span className="tab-icon">📷</span>
                <span className="tab-text">Camera</span>
              </button>
              <button 
                className={`tab ${activeTab === 'upload' ? 'active-tab' : ''}`} 
                onClick={() => setActiveTab('upload')}
              >
                <span className="tab-icon">📁</span>
                <span className="tab-text">Upload</span>
              </button>
            </div>

            <div className="panel-content">
              {activeTab === 'camera' && !loading && !preview && (
                <div className="camera-view">
                  <Webcam
                    audio={false}
                    ref={webcamRef}
                    screenshotFormat="image/jpeg"
                    videoConstraints={{ facingMode: "user" }}
                    className="webcam-preview"
                  />
                  <button className="camera-button" onClick={capture}>
                    Take Photo
                  </button>
                  <p className="help-text">Center your face in the frame and look straight ahead</p>
                </div>
              )}

              {activeTab === 'upload' && !loading && !preview && (
                <div className="upload-view">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={onUpload}
                    ref={fileInputRef}
                    className="file-input"
                  />
                  <div className="drop-area" onClick={triggerFileInput}>
                    <div className="upload-icon">📤</div>
                    <p className="upload-text">Click or drag photo here</p>
                    <p className="upload-help">Please use a front-facing portrait photo</p>
                  </div>
                </div>
              )}

              {loading && (
                <div className="loading-view">
                  <div className="spinner"></div>
                  <p className="loading-text">Analyzing your face shape...</p>
                  <p className="loading-subtext">This will just take a moment</p>
                </div>
              )}

              {preview && !loading && appState === 'results' && result && (
                <div className="results-view">
                  <div className="results-header">
                    <h3 className="results-title">Your Results</h3>
                    <div className="results-summary">
                      <div className="user-photo">
                        <img src={preview} alt="Your photo" className="photo-preview" />
                      </div>
                      <div className="results-details">
                        <div className="result-item">
                          <span className="result-label">Gender:</span>
                          <span className="result-value"> {result.gender? result.gender.charAt(0).toUpperCase() + result.gender.slice(1): ''}</span>
                        </div>
                        <div className="result-item">
                          <span className="result-label">Face Shape:</span>
                          <span className="result-value"> {result.shape? result.shape.charAt(0).toUpperCase() + result.shape.slice(1): ''}</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="recommendations">
                    <h3 className="recommendations-title">Recommended Hairstyles</h3>
                    <div className="styles-grid">
                      {getImagePaths().map((path, index) => (
                        <div key={index} className="style-card">
                          <div className="style-image-container">
                            <img src={path} alt={`Style ${index + 1}`} className="style-image" />
                          </div>
                          <div className="style-name">Style {index + 1}</div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <button className="try-again" onClick={resetApp}>
                    Try Another Photo
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}