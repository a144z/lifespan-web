import * as ort from 'onnxruntime-web';

export interface PredictionResult {
  lifespan: number;
  confidence?: number;
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
      // Configure ONNX Runtime for web
      ort.env.wasm.numThreads = 1; // Use single thread for better compatibility
      ort.env.wasm.simd = true; // Enable SIMD for faster inference

      console.log('Loading ONNX model from:', this.modelPath);
      this.session = await ort.InferenceSession.create(this.modelPath, {
        executionProviders: ['wasm'], // Use WebAssembly backend
        graphOptimizationLevel: 'all', // Enable all optimizations
      });

      this.isLoaded = true;
      console.log('✓ Model loaded successfully');
    } catch (error) {
      console.error('Error loading model:', error);
      throw new Error(`Failed to load model: ${error}`);
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

