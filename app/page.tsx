'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { LifespanPredictor } from '@/lib/onnxModel';
import { 
  createFaceDetector, 
  detectFaces, 
  cropFaceFromVideo, 
  drawFaceBox,
  type FaceBox
} from '@/lib/faceDetection';

type Mode = 'upload' | 'camera';

export default function Home() {
  const [mode, setMode] = useState<Mode>('upload');
  const [prediction, setPrediction] = useState<number | null>(null); // For upload mode
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [fps, setFps] = useState(0);
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('user'); // 'user' = front, 'environment' = back
  
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
  // Store predictions for each face (keyed by face position hash)
  // Map structure: faceKey -> prediction value, faceKey_time -> timestamp
  const facePredictionsRef = useRef<Map<string, number>>(new Map());
  const currentFacesRef = useRef<FaceBox[]>([]);

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

  // Ensure canvas dimensions match video dimensions, especially important on mobile
  // MediaPipe coordinates are relative to video.videoWidth/video.videoHeight
  // Canvas must match these exact dimensions for accurate coordinate mapping
  useEffect(() => {
    if (mode === 'camera' && videoRef.current && canvasRef.current) {
      const updateCanvasSize = () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (video && canvas && video.videoWidth > 0 && video.videoHeight > 0) {
          // Match canvas internal dimensions to video's ACTUAL dimensions
          // This ensures coordinate mapping is accurate regardless of CSS scaling
          // MediaPipe returns coordinates in video.videoWidth/video.videoHeight space
          if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
          }
        }
      };

      const video = videoRef.current;
      video.addEventListener('loadedmetadata', updateCanvasSize);
      video.addEventListener('resize', updateCanvasSize);
      video.addEventListener('loadeddata', updateCanvasSize);
      
      // Also check on video ready
      if (video.readyState >= video.HAVE_METADATA) {
        updateCanvasSize();
      }

      // Periodic check for mobile devices where dimensions might change
      const intervalId = setInterval(updateCanvasSize, 500);

      return () => {
        video.removeEventListener('loadedmetadata', updateCanvasSize);
        video.removeEventListener('resize', updateCanvasSize);
        video.removeEventListener('loadeddata', updateCanvasSize);
        clearInterval(intervalId);
      };
    }
  }, [mode, cameraActive]);

  const stopCamera = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    // Clear predictions and canvas when stopping camera
    setPrediction(null);
    facePredictionsRef.current.clear();
    currentFacesRef.current = [];
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }
    setCameraActive(false);
  }, []);

  // Helper function to generate a stable key for a face box
  const getFaceKey = useCallback((box: FaceBox): string => {
    // Round coordinates to create stable keys for similar face positions
    const roundedX = Math.round(box.x / 10) * 10;
    const roundedY = Math.round(box.y / 10) * 10;
    return `${roundedX},${roundedY},${Math.round(box.width)},${Math.round(box.height)}`;
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
      // Still draw the last detected faces even if we skip processing
      if (canvas && video && currentFacesRef.current.length > 0) {
        // Ensure canvas size matches video's ACTUAL dimensions (not CSS display size)
        // MediaPipe coordinates are relative to video.videoWidth/video.videoHeight
        if (video.videoWidth > 0 && video.videoHeight > 0) {
          if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
          }
        }
        
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          // Draw all faces with their predictions
          for (const faceBox of currentFacesRef.current) {
            const faceKey = getFaceKey(faceBox);
            const prediction = facePredictionsRef.current.get(faceKey);
            drawFaceBox(ctx, faceBox, prediction, facingMode === 'user');
          }
        }
      } else if (canvas && currentFacesRef.current.length === 0) {
        // Clear canvas if no faces detected
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
      }
      animationFrameRef.current = requestAnimationFrame(processVideoFrames);
      return;
    }
    lastFrameTimeRef.current = now;

    try {
      // Get video frame
      if (video.readyState === video.HAVE_ENOUGH_DATA) {
        // Detect all faces using MediaPipe
        const faceBoxes = await detectFaces(faceDetectorRef.current, video);
        
        // Draw bounding boxes on canvas
        if (canvas) {
          // Ensure canvas internal dimensions match video's ACTUAL dimensions
          // This is critical for accurate coordinate mapping, especially on mobile
          // MediaPipe returns coordinates relative to video.videoWidth/video.videoHeight
          // NOT relative to the CSS display size
          if (video.videoWidth > 0 && video.videoHeight > 0) {
            // Only update canvas size if it changed to avoid unnecessary redraws
            if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
              canvas.width = video.videoWidth;
              canvas.height = video.videoHeight;
            }
          }
          
          const ctx = canvas.getContext('2d');
          if (ctx) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Update current faces
            currentFacesRef.current = faceBoxes;
            
            // Draw all faces
            for (const faceBox of faceBoxes) {
              const faceKey = getFaceKey(faceBox);
              const existingPrediction = facePredictionsRef.current.get(faceKey);
              
              // Draw box with existing prediction (if available)
              drawFaceBox(ctx, faceBox, existingPrediction, facingMode === 'user');
              
              // Predict for this face (only if we don't have a prediction yet, or update periodically)
              // Update prediction every 1 second for each face to avoid too many predictions
              const timeKey = `${faceKey}_time`;
              const lastUpdateTime = facePredictionsRef.current.get(timeKey);
              const shouldUpdate = !existingPrediction || 
                (lastUpdateTime !== undefined && (now - lastUpdateTime > 1000));
              
              if (shouldUpdate) {
                // Predict asynchronously without blocking drawing
                (async () => {
                  try {
                    const faceImageData = cropFaceFromVideo(video, faceBox);
                    const result = await predictorRef.current!.predict(faceImageData);
                    facePredictionsRef.current.set(faceKey, result.lifespan);
                    facePredictionsRef.current.set(timeKey, now);
                    
                    // Redraw with new prediction
                    if (ctx && canvas) {
                      ctx.clearRect(0, 0, canvas.width, canvas.height);
                      for (const f of currentFacesRef.current) {
                        const fKey = getFaceKey(f);
                        const fPred = facePredictionsRef.current.get(fKey);
                        drawFaceBox(ctx, f, fPred, facingMode === 'user');
                      }
                    }
                  } catch (predErr) {
                    console.error('Prediction error:', predErr);
                  }
                })();
              }
            }
            
            // Clear predictions for faces that are no longer detected
            const currentKeys = new Set(faceBoxes.map(box => getFaceKey(box)));
            const keysToDelete: string[] = [];
            for (const [key] of facePredictionsRef.current.entries()) {
              if (!key.endsWith('_time') && !currentKeys.has(key)) {
                keysToDelete.push(key);
                keysToDelete.push(`${key}_time`);
              }
            }
            keysToDelete.forEach(key => facePredictionsRef.current.delete(key));
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
  }, [getFaceKey]);

  const startCamera = useCallback(async (facing: 'user' | 'environment' = facingMode) => {
    try {
      setError(null);
      
      // Stop existing camera if any
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
      
      // Request camera access with specified facing mode
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: facing // 'user' = front, 'environment' = back
        }
      });
      
      streamRef.current = stream;
      setFacingMode(facing);
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setCameraActive(true);
        
        // Clear previous predictions when switching cameras
        facePredictionsRef.current.clear();
        currentFacesRef.current = [];
        
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
  }, [processVideoFrames, facingMode]);

  const switchCamera = useCallback(async () => {
    if (!cameraActive) return;
    
    try {
      const newFacingMode = facingMode === 'user' ? 'environment' : 'user';
      await startCamera(newFacingMode);
    } catch (err) {
      // If switching fails (e.g., back camera not available), show error but keep current camera
      setError(`Failed to switch camera: ${err}. The ${facingMode === 'user' ? 'back' : 'front'} camera may not be available on this device.`);
    }
  }, [cameraActive, facingMode, startCamera]);

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
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-4 sm:py-8 md:py-12 px-3 sm:px-4 md:px-6">
      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-xl sm:rounded-2xl shadow-xl p-4 sm:p-6 md:p-8">
          <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold text-center text-gray-800 mb-2">
            Remaining Lifespan Prediction
          </h1>
          <p className="text-center text-gray-600 text-sm sm:text-base mb-4 sm:mb-6 md:mb-8 px-2">
            Upload a face image or use your camera for real-time prediction
          </p>

          {/* Mode Toggle */}
          <div className="flex justify-center mb-4 sm:mb-6">
            <div className="inline-flex rounded-lg border border-gray-300 p-1 bg-gray-100 w-full sm:w-auto">
              <button
                onClick={() => handleModeChange('upload')}
                className={`flex-1 sm:flex-none px-3 sm:px-6 py-2 text-sm sm:text-base rounded-md font-medium transition-colors ${
                  mode === 'upload'
                    ? 'bg-white text-indigo-600 shadow-sm'
                    : 'text-gray-600 hover:text-gray-800'
                }`}
              >
                <span className="hidden sm:inline">📷 </span>Upload Photo
              </button>
              <button
                onClick={() => handleModeChange('camera')}
                className={`flex-1 sm:flex-none px-3 sm:px-6 py-2 text-sm sm:text-base rounded-md font-medium transition-colors ${
                  mode === 'camera'
                    ? 'bg-white text-indigo-600 shadow-sm'
                    : 'text-gray-600 hover:text-gray-800'
                }`}
              >
                <span className="hidden sm:inline">🎥 </span>Live Camera
              </button>
            </div>
          </div>

          {!modelLoaded && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 sm:p-4 mb-4 sm:mb-6">
              <p className="text-yellow-800 text-sm sm:text-base">Loading AI model...</p>
            </div>
          )}

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3 sm:p-4 mb-4 sm:mb-6">
              <p className="text-red-800 text-sm sm:text-base break-words">{error}</p>
            </div>
          )}

          <div className="space-y-4 sm:space-y-6">
            {/* Camera Mode */}
            {mode === 'camera' && (
              <div className="space-y-3 sm:space-y-4">
                <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-full object-cover"
                    style={{ transform: facingMode === 'user' ? 'scaleX(-1)' : 'none' }} // Mirror only front camera
                  />
                  <canvas
                    ref={canvasRef}
                    className="absolute top-0 left-0 w-full h-full pointer-events-none"
                    style={{ 
                      transform: facingMode === 'user' ? 'scaleX(-1)' : 'none',
                      objectFit: 'cover' // Ensure canvas scales properly
                    }} // Mirror to match video (only for front camera)
                  />
                  {!cameraActive && (
                    <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-75">
                      <button
                        onClick={() => startCamera()}
                        disabled={!modelLoaded}
                        className="px-4 sm:px-8 py-3 sm:py-4 bg-indigo-600 text-white rounded-lg text-sm sm:text-base font-semibold hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                      >
                        {modelLoaded ? '🎥 Start Camera' : 'Loading Model...'}
                      </button>
                    </div>
                  )}
                  {cameraActive && (
                    <>
                      <div className="absolute top-2 sm:top-4 right-2 sm:right-4 flex gap-2">
                        {/* Switch Camera Button - Only show on mobile or when back camera is available */}
                        <button
                          onClick={switchCamera}
                          className="px-3 sm:px-4 py-1.5 sm:py-2 bg-indigo-600 text-white rounded-lg text-xs sm:text-sm font-semibold hover:bg-indigo-700 transition-colors flex items-center gap-1"
                          title={facingMode === 'user' ? 'Switch to back camera' : 'Switch to front camera'}
                        >
                          <span className="text-base sm:text-lg">
                            {facingMode === 'user' ? '📷' : '📱'}
                          </span>
                          <span className="hidden sm:inline">
                            {facingMode === 'user' ? 'Back' : 'Front'}
                          </span>
                        </button>
                        <button
                          onClick={stopCamera}
                          className="px-3 sm:px-4 py-1.5 sm:py-2 bg-red-600 text-white rounded-lg text-xs sm:text-sm font-semibold hover:bg-red-700 transition-colors"
                        >
                          Stop
                        </button>
                      </div>
                    </>
                  )}
                  {fps > 0 && (
                    <div className="absolute top-2 sm:top-4 left-2 sm:left-4 bg-black bg-opacity-50 text-white px-2 sm:px-3 py-1 rounded text-xs sm:text-sm">
                      {fps} FPS
                    </div>
                  )}
                </div>
                
                {cameraActive && (
                  <div className="text-center text-xs sm:text-sm text-gray-600 px-2">
                    <p>Face detection active - green box shows detected face</p>
                    <p className="text-xs mt-1">Predictions update in real-time when face is detected</p>
                  </div>
                )}
              </div>
            )}

            {/* Upload Mode */}
            {mode === 'upload' && (
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 sm:p-6 md:p-8 text-center hover:border-indigo-400 transition-colors">
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
                    className="mx-auto h-8 w-8 sm:h-12 sm:w-12 text-gray-400"
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
                  <span className="mt-2 block text-xs sm:text-sm font-medium text-gray-700">
                    {loading ? 'Processing...' : 'Click to upload face image'}
                  </span>
                </label>
              </div>
            )}

            {/* Image Preview */}
            {imagePreview && (
              <div className="flex justify-center">
                <div className="relative w-full max-w-md">
                  <img
                    src={imagePreview}
                    alt="Uploaded face"
                    className="w-full h-auto rounded-lg shadow-lg"
                  />
                  {loading && (
                    <div className="absolute inset-0 bg-black bg-opacity-50 rounded-lg flex items-center justify-center">
                      <div className="text-white text-sm sm:text-lg">Analyzing...</div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Prediction Result - Only show for upload mode */}
            {prediction !== null && mode === 'upload' && (
              <div className="bg-gradient-to-r from-indigo-500 to-purple-600 rounded-lg p-4 sm:p-6 md:p-8 text-center text-white">
                <h2 className="text-lg sm:text-xl md:text-2xl font-semibold mb-3 sm:mb-4">
                  Prediction Result
                </h2>
                <div className="text-4xl sm:text-5xl md:text-6xl font-bold mb-2">{prediction.toFixed(1)}</div>
                <p className="text-base sm:text-lg md:text-xl">years remaining</p>
                <button
                  onClick={handleReset}
                  className="mt-4 sm:mt-6 px-4 sm:px-6 py-2 bg-white text-indigo-600 rounded-lg text-sm sm:text-base font-semibold hover:bg-gray-100 transition-colors"
                >
                  Try Another Image
                </button>
              </div>
            )}
          </div>

          {/* Info Section */}
          <div className="mt-6 sm:mt-8 pt-6 sm:pt-8 border-t border-gray-200">
            <h3 className="text-base sm:text-lg font-semibold text-gray-800 mb-3 sm:mb-4">About</h3>
            <div className="space-y-3 sm:space-y-4 text-xs sm:text-sm text-gray-600">
              <p>
                This tool uses artificial intelligence to estimate how many years a person might have remaining based on facial features. 
                You can upload a photo or use your camera for real-time predictions.
              </p>
              
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 sm:p-4">
                <p className="font-semibold text-yellow-800 mb-2 text-xs sm:text-sm">⚠️ Important Disclaimer:</p>
                <ul className="space-y-1 text-yellow-700 text-xs leading-relaxed">
                  <li>• This prediction is <strong>not medically accurate</strong> and should not be used for health decisions</li>
                  <li>• The model was <strong>not trained on babies or children</strong> - results may not be meaningful for young people</li>
                  <li>• Maximum prediction is around <strong>50 years</strong> - the dataset used for training doesn't include longer lifespans</li>
                  <li>• Predictions of <strong>40-50 years</strong> indicate the person may live a long life based on patterns in the training data</li>
                  <li>• This is for entertainment and educational purposes only</li>
                </ul>
              </div>
              
              <p className="text-xs text-gray-500 leading-relaxed">
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
