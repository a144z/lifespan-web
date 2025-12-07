'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { LifespanPredictor } from '@/lib/onnxModel';
import { 
  createFaceDetector, 
  detectFace, 
  cropFaceFromVideo, 
  drawFaceBox,
  type FaceBox
} from '@/lib/faceDetection';

type Mode = 'upload' | 'camera';

export default function Home() {
  const [mode, setMode] = useState<Mode>('upload');
  const [prediction, setPrediction] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [fps, setFps] = useState(0);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const predictorRef = useRef<LifespanPredictor | null>(null);
  const faceDetectorRef = useRef<Awaited<ReturnType<typeof createFaceDetector>> | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef<number>(0);
  const frameCountRef = useRef<number>(0);
  const fpsUpdateTimeRef = useRef<number>(Date.now());
  const currentFaceBoxRef = useRef<FaceBox | null>(null);

  // Initialize model and face detector on component mount
  useEffect(() => {
    const initModels = async () => {
      try {
        // Initialize lifespan predictor
        const predictor = new LifespanPredictor('/models/lifespan_model_fastvit.onnx');
        await predictor.load();
        predictorRef.current = predictor;
        
        // Initialize face detector
        const faceDetector = await createFaceDetector();
        faceDetectorRef.current = faceDetector;
        
        setModelLoaded(true);
      } catch (err) {
        setError(`Failed to load models: ${err}`);
      }
    };
    initModels();
  }, []);

  // Cleanup camera on unmount or mode change
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [mode]);

  const stopCamera = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    // Clear prediction and canvas when stopping camera
    setPrediction(null);
    currentFaceBoxRef.current = null;
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }
    setCameraActive(false);
  }, []);

  const processVideoFrames = useCallback(async () => {
    if (!videoRef.current || !predictorRef.current || !faceDetectorRef.current || !streamRef.current) {
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    // Throttle to ~10 FPS for performance (process every 100ms)
    const now = Date.now();
    if (now - lastFrameTimeRef.current < 100) {
      // Still draw the last detected face box even if we skip processing
      if (canvas && currentFaceBoxRef.current) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          // Draw directly (CSS handles mirroring)
          drawFaceBox(ctx, currentFaceBoxRef.current);
        }
      } else if (canvas && !currentFaceBoxRef.current) {
        // Clear canvas and prediction if no face was detected
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
        setPrediction(null);
      }
      animationFrameRef.current = requestAnimationFrame(processVideoFrames);
      return;
    }
    lastFrameTimeRef.current = now;

    try {
      // Get video frame
      if (video.readyState === video.HAVE_ENOUGH_DATA) {
        // Detect face using MediaPipe
        const faceBox = await detectFace(faceDetectorRef.current, video);
        
        // Draw bounding box on canvas
        if (canvas) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          const ctx = canvas.getContext('2d');
          if (ctx) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            if (faceBox) {
              // Draw box directly (Canvas is already mirrored via CSS to match Video)
              drawFaceBox(ctx, faceBox);
              
              currentFaceBoxRef.current = faceBox;
              
              // Crop face and predict
              try {
                // This ensures ONLY the face region inside the box is sent to the AI
                const faceImageData = cropFaceFromVideo(video, faceBox);
                const result = await predictorRef.current!.predict(faceImageData);
                setPrediction(result.lifespan);
              } catch (predErr) {
                console.error('Prediction error:', predErr);
              }
            } else {
              currentFaceBoxRef.current = null;
              setPrediction(null);
            }
          }
        }
        
        // Update FPS counter
        frameCountRef.current++;
        if (now - fpsUpdateTimeRef.current > 1000) {
          setFps(frameCountRef.current);
          frameCountRef.current = 0;
          fpsUpdateTimeRef.current = now;
        }
      }
    } catch (err) {
      console.error('Frame processing error:', err);
    }
    
    // Continue processing
    if (streamRef.current) {
      animationFrameRef.current = requestAnimationFrame(processVideoFrames);
    }
  }, []);

  const startCamera = useCallback(async () => {
    try {
      setError(null);
      
      // Request camera access
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user' // Front camera
        }
      });
      
      streamRef.current = stream;
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setCameraActive(true);
        
        // Wait for video to be ready, then start processing
        videoRef.current.onloadedmetadata = async () => {
          try {
            await videoRef.current?.play();
            // Small delay to ensure video is playing
            setTimeout(() => {
              processVideoFrames();
            }, 100);
          } catch (err) {
            console.error('Video play error:', err);
          }
        };
      }
    } catch (err) {
      setError(`Camera access denied: ${err}`);
      setCameraActive(false);
    }
  }, [processVideoFrames]);

  const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file');
      return;
    }

    setError(null);
    setLoading(true);
    setPrediction(null);

    try {
      // Create image preview
      const reader = new FileReader();
      reader.onload = async (e) => {
        const imageUrl = e.target?.result as string;
        setImagePreview(imageUrl);

        // Create image element
        const img = new Image();
        img.onload = async () => {
          try {
            // Create canvas to get ImageData
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');
            if (!ctx) {
              throw new Error('Could not get canvas context');
            }
            ctx.drawImage(img, 0, 0);
            const imageData = ctx.getImageData(0, 0, img.width, img.height);

            // Predict
            if (!predictorRef.current) {
              const predictor = new LifespanPredictor('/models/lifespan_model_fastvit.onnx');
              await predictor.load();
              predictorRef.current = predictor;
            }

            const result = await predictorRef.current.predict(imageData);
            setPrediction(result.lifespan);
          } catch (err) {
            setError(`Prediction failed: ${err}`);
          } finally {
            setLoading(false);
          }
        };
        img.onerror = () => {
          setError('Failed to load image');
          setLoading(false);
        };
        img.src = imageUrl;
      };
      reader.readAsDataURL(file);
    } catch (err) {
      setError(`Error processing image: ${err}`);
      setLoading(false);
    }
  };

  const handleReset = () => {
    setPrediction(null);
    setImagePreview(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleModeChange = (newMode: Mode) => {
    if (newMode === 'camera' && mode === 'upload') {
      // Switching to camera mode
      setMode('camera');
      setPrediction(null);
      setImagePreview(null);
    } else if (newMode === 'upload' && mode === 'camera') {
      // Switching to upload mode
      stopCamera();
      setMode('upload');
      setPrediction(null);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-12 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-2xl shadow-xl p-8">
          <h1 className="text-4xl font-bold text-center text-gray-800 mb-2">
            Remaining Lifespan Prediction
          </h1>
          <p className="text-center text-gray-600 mb-8">
            Upload a face image or use your camera for real-time prediction
          </p>

          {/* Mode Toggle */}
          <div className="flex justify-center mb-6">
            <div className="inline-flex rounded-lg border border-gray-300 p-1 bg-gray-100">
              <button
                onClick={() => handleModeChange('upload')}
                className={`px-6 py-2 rounded-md font-medium transition-colors ${
                  mode === 'upload'
                    ? 'bg-white text-indigo-600 shadow-sm'
                    : 'text-gray-600 hover:text-gray-800'
                }`}
              >
                📷 Upload Photo
              </button>
              <button
                onClick={() => handleModeChange('camera')}
                className={`px-6 py-2 rounded-md font-medium transition-colors ${
                  mode === 'camera'
                    ? 'bg-white text-indigo-600 shadow-sm'
                    : 'text-gray-600 hover:text-gray-800'
                }`}
              >
                🎥 Live Camera
              </button>
            </div>
          </div>

          {!modelLoaded && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
              <p className="text-yellow-800">Loading AI model...</p>
            </div>
          )}

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
              <p className="text-red-800">{error}</p>
            </div>
          )}

          <div className="space-y-6">
            {/* Camera Mode */}
            {mode === 'camera' && (
              <div className="space-y-4">
                <div className="relative bg-black rounded-lg overflow-hidden">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full max-w-2xl mx-auto"
                    style={{ transform: 'scaleX(-1)' }} // Mirror effect
                  />
                  <canvas
                    ref={canvasRef}
                    className="absolute top-0 left-0 w-full h-full pointer-events-none"
                    style={{ transform: 'scaleX(-1)' }} // Mirror to match video
                  />
                  {!cameraActive && (
                    <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-75">
                      <button
                        onClick={startCamera}
                        disabled={!modelLoaded}
                        className="px-8 py-4 bg-indigo-600 text-white rounded-lg font-semibold hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                      >
                        {modelLoaded ? '🎥 Start Camera' : 'Loading Model...'}
                      </button>
                    </div>
                  )}
                  {cameraActive && (
                    <div className="absolute top-4 right-4">
                      <button
                        onClick={stopCamera}
                        className="px-4 py-2 bg-red-600 text-white rounded-lg font-semibold hover:bg-red-700 transition-colors"
                      >
                        Stop Camera
                      </button>
                    </div>
                  )}
                  {fps > 0 && (
                    <div className="absolute top-4 left-4 bg-black bg-opacity-50 text-white px-3 py-1 rounded">
                      {fps} FPS
                    </div>
                  )}
                </div>
                
                {cameraActive && (
                  <div className="text-center text-sm text-gray-600">
                    <p>Face detection active - green box shows detected face</p>
                    <p className="text-xs mt-1">Predictions update in real-time when face is detected</p>
                  </div>
                )}
              </div>
            )}

            {/* Upload Mode */}
            {mode === 'upload' && (
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-indigo-400 transition-colors">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="hidden"
                  id="image-upload"
                  disabled={!modelLoaded || loading}
                />
                <label
                  htmlFor="image-upload"
                  className={`cursor-pointer block ${
                    !modelLoaded || loading ? 'opacity-50 cursor-not-allowed' : ''
                  }`}
                >
                  <svg
                    className="mx-auto h-12 w-12 text-gray-400"
                    stroke="currentColor"
                    fill="none"
                    viewBox="0 0 48 48"
                  >
                    <path
                      d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                      strokeWidth={2}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  <span className="mt-2 block text-sm font-medium text-gray-700">
                    {loading ? 'Processing...' : 'Click to upload face image'}
                  </span>
                </label>
              </div>
            )}

            {/* Image Preview */}
            {imagePreview && (
              <div className="flex justify-center">
                <div className="relative">
                  <img
                    src={imagePreview}
                    alt="Uploaded face"
                    className="max-w-md rounded-lg shadow-lg"
                  />
                  {loading && (
                    <div className="absolute inset-0 bg-black bg-opacity-50 rounded-lg flex items-center justify-center">
                      <div className="text-white text-lg">Analyzing...</div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Prediction Result */}
            {prediction !== null && (
              <div className="bg-gradient-to-r from-indigo-500 to-purple-600 rounded-lg p-8 text-center text-white">
                <h2 className="text-2xl font-semibold mb-4">
                  {mode === 'camera' ? 'Live Prediction' : 'Prediction Result'}
                </h2>
                <div className="text-6xl font-bold mb-2">{prediction.toFixed(1)}</div>
                <p className="text-xl">years remaining</p>
                {mode === 'upload' && (
                  <button
                    onClick={handleReset}
                    className="mt-6 px-6 py-2 bg-white text-indigo-600 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
                  >
                    Try Another Image
                  </button>
                )}
                {mode === 'camera' && (
                  <p className="mt-4 text-sm opacity-90">
                    Prediction updates automatically as you move
                  </p>
                )}
              </div>
            )}
          </div>

          {/* Info Section */}
          <div className="mt-8 pt-8 border-t border-gray-200">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">About</h3>
            <div className="space-y-4 text-sm text-gray-600">
              <p>
                This tool uses artificial intelligence to estimate how many years a person might have remaining based on facial features. 
                You can upload a photo or use your camera for real-time predictions.
              </p>
              
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <p className="font-semibold text-yellow-800 mb-2">⚠️ Important Disclaimer:</p>
                <ul className="space-y-1 text-yellow-700 text-xs">
                  <li>• This prediction is <strong>not medically accurate</strong> and should not be used for health decisions</li>
                  <li>• The model was <strong>not trained on babies or children</strong> - results may not be meaningful for young people</li>
                  <li>• Maximum prediction is around <strong>50 years</strong> - the dataset used for training doesn't include longer lifespans</li>
                  <li>• Predictions of <strong>40-50 years</strong> indicate the person may live a long life based on patterns in the training data</li>
                  <li>• This is for entertainment and educational purposes only</li>
                </ul>
              </div>
              
              <p className="text-xs text-gray-500">
                All processing happens in your browser - your photos are never sent to any server. 
                Your privacy is completely protected.
              </p>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
