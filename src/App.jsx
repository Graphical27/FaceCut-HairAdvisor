import { useState, useRef, useEffect } from 'react'
import { Camera, Upload, ArrowLeft, Sun, Moon } from 'lucide-react'

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
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData
      })
      const data = await res.json()
      setResult(data)
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
    if (!webcamRef.current) return;
    
    // Create canvas to capture the video frame
    const video = webcamRef.current;
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw the video frame to canvas (flip it back to normal)
    context.scale(-1, 1);
    context.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
    
    // Convert to blob
    canvas.toBlob(async (blob) => {
      if (blob) {
        const imageSrc = URL.createObjectURL(blob);
        setPreview(imageSrc);
        const file = new File([blob], 'selfie.jpg', { type: 'image/jpeg' });
        sendFile(file);
      }
    }, 'image/jpeg', 0.8);
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

  // Webcam component (assuming react-webcam is available in your environment)
  const Webcam = ({ audio, ref, screenshotFormat, videoConstraints, className }) => {
    useEffect(() => {
      // Request camera permissions
      navigator.mediaDevices.getUserMedia({ video: videoConstraints })
        .then(stream => {
          if (ref.current) {
            ref.current.srcObject = stream;
          }
        })
        .catch(err => console.error('Camera access denied:', err));
    }, []);

    return (
      <video
        ref={ref}
        autoPlay
        playsInline
        muted
        className={className}
        style={{ transform: 'scaleX(-1)' }} // Mirror the video
      />
    );
  };

  return (
    <div className={`min-h-screen transition-colors duration-300 ${darkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'}`}>
      {/* Theme Toggle */}
      <div className="fixed top-4 right-4 z-50">
        <button
          className={`p-3 rounded-full shadow-lg transition-all duration-300 ${
            darkMode 
              ? 'bg-gray-800 hover:bg-gray-700 text-yellow-400' 
              : 'bg-white hover:bg-gray-100 text-gray-600'
          }`}
          onClick={toggleDarkMode}
          title={darkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
        >
          {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
        </button>
      </div>

      {/* Home Screen */}
      {appState === 'home' && (
        <div className="home-screen" ref={homeRef}>
          <div className="container mx-auto px-6 py-12">
            <div className="text-center mb-12">
              <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full mb-6">
                <span className="text-3xl">✂️</span>
              </div>
              <h1 className="text-6xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent mb-4">
                Perfect Cut
              </h1>
              <p className="text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
                Find your ideal hairstyle based on your face shape using advanced AI technology
              </p>
            </div>

            {/* Features Grid */}
            <div className="grid md:grid-cols-3 gap-8 mb-12">
              <div className={`p-6 rounded-2xl shadow-lg transition-all duration-300 hover:scale-105 ${
                darkMode ? 'bg-gray-800' : 'bg-white'
              }`}>
                <div className="text-4xl mb-4">📷</div>
                <h3 className="text-xl font-semibold mb-2">Face Analysis</h3>
                <p className="text-gray-600 dark:text-gray-300">
                  Advanced AI technology to analyze your face shape with precision
                </p>
              </div>
              <div className={`p-6 rounded-2xl shadow-lg transition-all duration-300 hover:scale-105 ${
                darkMode ? 'bg-gray-800' : 'bg-white'
              }`}>
                <div className="text-4xl mb-4">💇</div>
                <h3 className="text-xl font-semibold mb-2">Style Matching</h3>
                <p className="text-gray-600 dark:text-gray-300">
                  Get personalized hairstyle recommendations tailored to you
                </p>
              </div>
              <div className={`p-6 rounded-2xl shadow-lg transition-all duration-300 hover:scale-105 ${
                darkMode ? 'bg-gray-800' : 'bg-white'
              }`}>
                <div className="text-4xl mb-4">🔄</div>
                <h3 className="text-xl font-semibold mb-2">Instant Results</h3>
                <p className="text-gray-600 dark:text-gray-300">
                  See your perfect hairstyle matches in seconds
                </p>
              </div>
            </div>

            {/* CTA Button */}
            <div className="text-center mb-12">
              <button
                className="inline-flex items-center px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white font-semibold rounded-full shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-300"
                onClick={startAnalyzer}
              >
                <span className="mr-2">Find My Perfect Hairstyle</span>
                <ArrowLeft className="w-5 h-5 rotate-180" />
              </button>
            </div>

            {/* Testimonial */}
            <div className="text-center">
              <div className={`inline-block p-6 rounded-2xl ${
                darkMode ? 'bg-gray-800' : 'bg-white'
              } shadow-lg max-w-md`}>
                <p className="text-lg italic mb-2">
                  "This app helped me find the perfect hairstyle for my face shape!"
                </p>
                <p className="text-sm font-semibold text-purple-600">- Alex S.</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Analyzer Screen */}
      {analyzerVisible && (
        <div className="analyzer min-h-screen" ref={analyzerRef}>
          <div className="container mx-auto px-6 py-8">
            {/* Back Button */}
            <button
              className={`inline-flex items-center px-4 py-2 rounded-lg mb-8 transition-colors ${
                darkMode 
                  ? 'bg-gray-800 hover:bg-gray-700 text-white' 
                  : 'bg-white hover:bg-gray-100 text-gray-900'
              } shadow-md`}
              onClick={resetApp}
            >
              <ArrowLeft className="w-5 h-5 mr-2" />
              <span>Back to Home</span>
            </button>

            {/* Analyzer Panel */}
            <div className={`max-w-4xl mx-auto rounded-2xl shadow-xl overflow-hidden ${
              darkMode ? 'bg-gray-800' : 'bg-white'
            }`}>
              <div className="p-8">
                <div className="text-center mb-8">
                  <h2 className="text-3xl font-bold mb-2">Face Shape Analyzer</h2>
                  <p className="text-gray-600 dark:text-gray-300">
                    Let's find the perfect hairstyle for your face shape
                  </p>
                </div>

                {/* Tabs */}
                <div className="flex justify-center mb-8">
                  <div className={`inline-flex rounded-lg p-1 ${
                    darkMode ? 'bg-gray-700' : 'bg-gray-100'
                  }`}>
                    <button
                      className={`inline-flex items-center px-6 py-3 rounded-md font-medium transition-all ${
                        activeTab === 'camera'
                          ? 'bg-purple-600 text-white shadow-md'
                          : darkMode 
                            ? 'text-gray-300 hover:text-white' 
                            : 'text-gray-500 hover:text-gray-700'
                      }`}
                      onClick={() => setActiveTab('camera')}
                    >
                      <Camera className="w-5 h-5 mr-2" />
                      <span>Camera</span>
                    </button>
                    <button
                      className={`inline-flex items-center px-6 py-3 rounded-md font-medium transition-all ${
                        activeTab === 'upload'
                          ? 'bg-purple-600 text-white shadow-md'
                          : darkMode 
                            ? 'text-gray-300 hover:text-white' 
                            : 'text-gray-500 hover:text-gray-700'
                      }`}
                      onClick={() => setActiveTab('upload')}
                    >
                      <Upload className="w-5 h-5 mr-2" />
                      <span>Upload</span>
                    </button>
                  </div>
                </div>

                {/* Content */}
                <div className="min-h-96 flex items-center justify-center">
                  {activeTab === 'camera' && !loading && !preview && (
                    <div className="text-center w-full max-w-md">
                      <Webcam 
                        audio={false}
                        ref={webcamRef}
                        screenshotFormat="image/jpeg"
                        videoConstraints={{ facingMode: "user" }}
                        className="w-full aspect-video rounded-lg mb-6 bg-gray-100 dark:bg-gray-700"
                      />
                      <button
                        className="inline-flex items-center px-6 py-3 bg-purple-600 text-white rounded-lg font-medium hover:bg-purple-700 transition-colors mb-4"
                        onClick={capture}
                      >
                        <Camera className="w-5 h-5 mr-2" />
                        Take Photo
                      </button>
                      <p className="text-sm text-gray-600 dark:text-gray-300">
                        Center your face in the frame and look straight ahead
                      </p>
                    </div>
                  )}

                  {activeTab === 'upload' && !loading && !preview && (
                    <div className="text-center w-full max-w-md">
                      <input
                        type="file"
                        accept="image/*"
                        onChange={onUpload}
                        ref={fileInputRef}
                        className="hidden"
                      />
                      <div
                        className={`border-2 border-dashed rounded-lg p-12 cursor-pointer transition-colors ${
                          darkMode 
                            ? 'border-gray-600 hover:border-purple-500 bg-gray-700 hover:bg-gray-600' 
                            : 'border-gray-300 hover:border-purple-500 bg-gray-50 hover:bg-gray-100'
                        }`}
                        onClick={triggerFileInput}
                      >
                        <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                        <p className="text-lg font-medium mb-2">Click or drag photo here</p>
                        <p className="text-sm text-gray-600 dark:text-gray-300">
                          Please use a front-facing portrait photo
                        </p>
                      </div>
                    </div>
                  )}

                  {loading && (
                    <div className="text-center">
                      <div className="w-16 h-16 border-4 border-purple-200 border-t-purple-600 rounded-full animate-spin mx-auto mb-4"></div>
                      <p className="text-lg font-medium mb-2">Analyzing your face shape...</p>
                      <p className="text-sm text-gray-600 dark:text-gray-300">
                        This will just take a moment
                      </p>
                    </div>
                  )}

                  {preview && !loading && appState === 'results' && result && (
                    <div className="w-full">
                      <div className="text-center mb-8">
                        <h3 className="text-2xl font-bold mb-6">Your Results</h3>
                        <div className="flex flex-col md:flex-row items-center justify-center gap-8 mb-8">
                          <div className="flex-shrink-0">
                            <img 
                              src={preview} 
                              alt="Your photo" 
                              className="w-32 h-32 rounded-full object-cover shadow-lg"
                            />
                          </div>
                          <div className="space-y-4">
                            <div className="flex items-center">
                              <span className="font-medium mr-2">Gender:</span>
                              <span className="px-3 py-1 bg-purple-100 dark:bg-purple-900 text-purple-800 dark:text-purple-200 rounded-full">
                                {result.gender ? result.gender.charAt(0).toUpperCase() + result.gender.slice(1) : ''}
                              </span>
                            </div>
                            <div className="flex items-center">
                              <span className="font-medium mr-2">Face Shape:</span>
                              <span className="px-3 py-1 bg-pink-100 dark:bg-pink-900 text-pink-800 dark:text-pink-200 rounded-full">
                                {result.shape ? result.shape.charAt(0).toUpperCase() + result.shape.slice(1) : ''}
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Recommendations */}
                      <div className="mb-8">
                        <h3 className="text-xl font-bold text-center mb-6">Recommended Hairstyles</h3>
                        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                          {getImagePaths().map((path, index) => (
                            <div key={index} className={`rounded-lg overflow-hidden shadow-md ${
                              darkMode ? 'bg-gray-700' : 'bg-gray-50'
                            }`}>
                              <div className="aspect-square bg-gray-200 dark:bg-gray-600 flex items-center justify-center">
                                <img 
                                  src={path} 
                                  alt={`Style ${index + 1}`} 
                                  className="w-full h-full object-cover"
                                  onError={(e) => {
                                    e.target.style.display = 'none'
                                    e.target.nextSibling.style.display = 'flex'
                                  }}
                                />
                                <div className="hidden w-full h-full items-center justify-center text-gray-400">
                                  <span className="text-2xl">💇</span>
                                </div>
                              </div>
                              <div className="p-3 text-center">
                                <span className="text-sm font-medium">Style {index + 1}</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div className="text-center">
                        <button
                          className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg font-medium hover:shadow-lg transform hover:scale-105 transition-all duration-300"
                          onClick={resetApp}
                        >
                          Try Another Photo
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <style jsx>{`
        .fade-out {
          opacity: 0;
          transform: translateY(-20px);
          transition: all 0.5s ease-in-out;
        }
        
        .analyzer-appear {
          opacity: 1;
          transform: translateY(0);
          transition: all 0.5s ease-in-out;
        }
      `}</style>
    </div>
  )
}
