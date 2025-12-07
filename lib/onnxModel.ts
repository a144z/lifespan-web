import * as ort from 'onnxruntime-web';

export interface PredictionResult {
  lifespan: number;
  confidence?: number;
}

// Initialize WASM backend once globally
let wasmInitialized = false;
let wasmInitPromise: Promise<void> | null = null;

async function initializeWasm(): Promise<void> {
  if (wasmInitialized || typeof window === 'undefined') {
    return;
  }

  // If initialization is already in progress, wait for it
  if (wasmInitPromise) {
    return wasmInitPromise;
  }

  wasmInitPromise = (async () => {
    try {
      // Configure WASM settings
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = false;
      ort.env.wasm.proxy = false;
      ort.env.wasm.initTimeout = 10000; // 10 second timeout
      
      // Use local WASM files first (from public/wasm), fallback to CDN
      // Local files are more reliable and faster
      if (typeof window !== 'undefined' && window.location) {
        const basePath = window.location.origin;
        ort.env.wasm.wasmPaths = `${basePath}/wasm/`;
      } else {
        // Fallback to CDN if window is not available
        ort.env.wasm.wasmPaths = 'https://unpkg.com/onnxruntime-web@1.14.0/dist/';
      }
      
      // Try to initialize WASM explicitly using wasmBackend
      try {
        // Try using wasmBackend if available
        const wasmBackend = (ort as any).wasmBackend;
        if (wasmBackend && typeof wasmBackend.init === 'function') {
          await wasmBackend.init({
            wasm: {
              numThreads: 1,
              simd: false,
              proxy: false,
            }
          });
          wasmInitialized = true;
          console.log('✓ WASM backend initialized');
        } else {
          // Mark as initialized - will auto-init on first session create
          wasmInitialized = true;
          console.log('✓ WASM configured, will auto-initialize on first session');
        }
      } catch (initError) {
        console.warn('WASM explicit init failed, will auto-initialize:', initError);
        // Continue - WASM might auto-initialize when creating a session
        wasmInitialized = true;
      }
    } catch (error) {
      console.error('WASM initialization error:', error);
      wasmInitPromise = null;
      // Don't throw - let it try to auto-initialize
      wasmInitialized = true;
    }
  })();

  return wasmInitPromise;
}

export class LifespanPredictor {
  private session: ort.InferenceSession | null = null;
  private modelPath: string;
  private isLoaded: boolean = false;

  constructor(modelPath: string = '/models/lifespan_model.onnx') {
    this.modelPath = modelPath;
  }

  async load(): Promise<void> {
    if (this.isLoaded && this.session) {
      return;
    }

    try {
      // Initialize WASM backend first
      await initializeWasm();

      // Ensure we use an absolute URL for the model path
      let modelUrl = this.modelPath;
      if (typeof window !== 'undefined' && window.location) {
        // If path is relative, make it absolute
        if (modelUrl.startsWith('/')) {
          modelUrl = `${window.location.origin}${modelUrl}`;
        } else if (!modelUrl.startsWith('http')) {
          modelUrl = `${window.location.origin}/${modelUrl}`;
        }
      }

      console.log('Loading ONNX model from:', modelUrl);
      
      // Verify the file exists by trying to fetch it first (for better error messages)
      if (typeof window !== 'undefined') {
        try {
          const response = await fetch(modelUrl, { method: 'HEAD' });
          if (!response.ok) {
            throw new Error(`Model file not found (HTTP ${response.status}): ${modelUrl}`);
          }
          console.log('✓ Model file found, size:', response.headers.get('content-length') || 'unknown');
        } catch (fetchError: any) {
          // If HEAD fails, try GET to see if it's a CORS issue
          if (fetchError.message && !fetchError.message.includes('HTTP')) {
            console.warn('HEAD request failed, model might still be accessible:', fetchError.message);
          } else {
            throw fetchError;
          }
        }
      }
      
      // Create session - WASM should now be initialized
      this.session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['wasm'], // Use WebAssembly backend
        graphOptimizationLevel: 'all', // Enable all optimizations
      });

      this.isLoaded = true;
      console.log('✓ Model loaded successfully');
    } catch (error: any) {
      console.error('Error loading model:', error);
      const errorMessage = error?.message || String(error);
      
      // Provide more helpful error messages
      if (errorMessage.includes('404') || errorMessage.includes('Failed to fetch')) {
        throw new Error(`Model file not found at ${this.modelPath}. Please ensure the model file exists in public/models/`);
      } else if (errorMessage.includes('protobuf') || errorMessage.includes('parsing')) {
        throw new Error(`Model file is corrupted or invalid. Please check the ONNX model file.`);
      } else {
        throw new Error(`Failed to load model: ${errorMessage}`);
      }
    }
  }

  async predict(imageData: ImageData): Promise<PredictionResult> {
    if (!this.session || !this.isLoaded) {
      await this.load();
    }

    if (!this.session) {
      throw new Error('Model not loaded');
    }

    try {
      // Preprocess image: resize to 224x224, normalize
      const preprocessed = this.preprocessImage(imageData);

      // Create tensor
      const tensor = new ort.Tensor('float32', preprocessed, [1, 3, 224, 224]);

      // Run inference
      const feeds = { input: tensor };
      const results = await this.session.run(feeds);
      const output = results.output;

      // Get prediction
      const lifespan = output.data[0] as number;

      return {
        lifespan: Math.max(0, lifespan), // Ensure non-negative
      };
    } catch (error) {
      console.error('Prediction error:', error);
      throw new Error(`Prediction failed: ${error}`);
    }
  }

  private preprocessImage(imageData: ImageData): Float32Array {
    // Create canvas for resizing
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) {
      throw new Error('Could not get canvas context');
    }

    // Create source canvas
    const srcCanvas = document.createElement('canvas');
    srcCanvas.width = imageData.width;
    srcCanvas.height = imageData.height;
    const srcCtx = srcCanvas.getContext('2d');
    if (!srcCtx) {
      throw new Error('Could not get source canvas context');
    }
    srcCtx.putImageData(imageData, 0, 0);

    // Draw and resize image
    ctx.drawImage(srcCanvas, 0, 0, 224, 224);

    // Get pixel data
    const resizedData = ctx.getImageData(0, 0, 224, 224);
    const pixels = resizedData.data;

    // Convert to CHW format and normalize (ImageNet stats)
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    const preprocessed = new Float32Array(3 * 224 * 224);

    for (let i = 0; i < 224 * 224; i++) {
      const r = pixels[i * 4] / 255.0;
      const g = pixels[i * 4 + 1] / 255.0;
      const b = pixels[i * 4 + 2] / 255.0;

      // Normalize
      preprocessed[i] = (r - mean[0]) / std[0]; // R channel
      preprocessed[224 * 224 + i] = (g - mean[1]) / std[1]; // G channel
      preprocessed[2 * 224 * 224 + i] = (b - mean[2]) / std[2]; // B channel
    }

    return preprocessed;
  }

  dispose(): void {
    if (this.session) {
      // ONNX.js doesn't have explicit dispose, but we can clear the reference
      this.session = null;
      this.isLoaded = false;
    }
  }
}

