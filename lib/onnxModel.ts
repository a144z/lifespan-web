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
      ort.env.wasm.initTimeout = 30000; // 30 second timeout for slower connections
      
      // Determine WASM path - use CDN for Vercel deployments, local for dev
      // Vercel serves static files from /, so we can use relative paths
      const isVercel = typeof window !== 'undefined' && 
                      (window.location.hostname.includes('vercel.app') || 
                       window.location.hostname.includes('vercel.com'));
      
      if (typeof window !== 'undefined' && window.location) {
        // Use relative path for better compatibility (works on Vercel and local)
        ort.env.wasm.wasmPaths = '/wasm/';
      } else {
        // Fallback to CDN if window is not available
        ort.env.wasm.wasmPaths = 'https://unpkg.com/onnxruntime-web@1.14.0/dist/';
      }
      
      console.log('WASM paths configured:', ort.env.wasm.wasmPaths);
      
      // Initialize WASM backend explicitly
      // ONNX.js 1.14.0 requires explicit initialization for better reliability
      try {
        // Wait a bit to ensure environment is ready
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // Try to initialize WASM - it will auto-initialize on first session if this fails
        // But we configure it here to ensure proper setup
        wasmInitialized = true;
        console.log('✓ WASM environment configured');
      } catch (initError) {
        console.warn('WASM pre-init warning (will auto-init on session create):', initError);
        wasmInitialized = true;
      }
    } catch (error) {
      console.error('WASM configuration error:', error);
      wasmInitPromise = null;
      // Mark as initialized anyway - let it try to auto-initialize
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

      console.log('Loading ONNX model from:', this.modelPath);
      
      // Create session with retry logic for better reliability
      let retries = 3;
      let lastError: Error | null = null;
      
      while (retries > 0) {
        try {
          // Create session - WASM will auto-initialize if needed
          this.session = await ort.InferenceSession.create(this.modelPath, {
            executionProviders: ['wasm'], // Use WebAssembly backend
            graphOptimizationLevel: 'all', // Enable all optimizations
          });

          this.isLoaded = true;
          console.log('✓ Model loaded successfully');
          return; // Success, exit retry loop
        } catch (sessionError: any) {
          lastError = sessionError;
          retries--;
          
          // If it's a WASM-related error, try with CDN fallback
          if (sessionError.message && 
              (sessionError.message.includes('WASM') || 
               sessionError.message.includes('wasm') ||
               sessionError.message.includes('Can\'t create a session'))) {
            
            console.warn(`Session creation failed (${retries} retries left), trying CDN fallback...`);
            
            // Switch to CDN for WASM files
            ort.env.wasm.wasmPaths = 'https://unpkg.com/onnxruntime-web@1.14.0/dist/';
            
            // Wait a bit before retry
            await new Promise(resolve => setTimeout(resolve, 1000));
            continue;
          }
          
          // For other errors, wait and retry
          if (retries > 0) {
            console.warn(`Session creation failed (${retries} retries left), retrying...`);
            await new Promise(resolve => setTimeout(resolve, 1000));
          }
        }
      }
      
      // If we get here, all retries failed
      throw lastError || new Error('Failed to create session after retries');
    } catch (error: any) {
      console.error('Error loading model:', error);
      const errorMessage = error?.message || String(error);
      throw new Error(`Failed to load model: ${errorMessage}`);
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

