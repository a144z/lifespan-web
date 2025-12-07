import { FaceDetector, FilesetResolver, Detection } from '@mediapipe/tasks-vision';

export interface FaceBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

let faceDetector: FaceDetector | null = null;

/**
 * Initialize MediaPipe Face Detector
 */
export async function createFaceDetector(): Promise<FaceDetector> {
  if (faceDetector) return faceDetector;

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  
  faceDetector = await FaceDetector.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite`,
      delegate: "GPU"
    },
    runningMode: "VIDEO"
  });
  
  return faceDetector;
}

/**
 * Detect face in video frame using MediaPipe
 * Returns a tight bounding box that perfectly fits the detected face
 */
export async function detectFace(
  detector: FaceDetector,
  video: HTMLVideoElement
): Promise<FaceBox | null> {
  const startTimeMs = performance.now();
  const detections: Detection[] = detector.detectForVideo(video, startTimeMs).detections;
  
  if (detections.length > 0) {
    // Get the first face (highest confidence)
    const detection = detections[0];
    const box = detection.boundingBox;
    
    if (box) {
      // Use MediaPipe's bounding box directly for perfect alignment
      // MediaPipe already provides accurate face boundaries
      return {
        x: box.originX,
        y: box.originY,
        width: box.width,
        height: box.height
      };
    }
  }
  
  return null;
}

/**
 * Draw face bounding box on canvas
 * Note: Canvas is mirrored via CSS, so text needs to be reverse-mirrored
 */
export function drawFaceBox(
  ctx: CanvasRenderingContext2D,
  box: FaceBox
) {
  const canvasWidth = ctx.canvas.width;
  
  ctx.strokeStyle = '#00ff00';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.roundRect(box.x, box.y, box.width, box.height, 10);
  ctx.stroke();
  
  // Add label - reverse mirror the text so it appears correctly
  ctx.save();
  ctx.scale(-1, 1);
  ctx.translate(-canvasWidth, 0);
  ctx.fillStyle = '#00ff00';
  ctx.font = '16px sans-serif';
  // Calculate mirrored x position for text
  const textX = canvasWidth - box.x;
  ctx.fillText('Face Detected', textX, box.y - 10);
  ctx.restore();
}

/**
 * Crop face from video element directly.
 * This extracts ONLY the pixels within the faceBox from the video frame.
 * Uses the exact faceBox coordinates for perfect alignment.
 * The result is resized to 224x224 for the AI model.
 */
export function cropFaceFromVideo(
  video: HTMLVideoElement,
  faceBox: FaceBox
): ImageData {
  const canvas = document.createElement('canvas');
  // AI model expects 224x224 input
  canvas.width = 224; 
  canvas.height = 224;
  const ctx = canvas.getContext('2d');
  
  if (!ctx) {
    throw new Error('Could not get canvas context');
  }
  
  // Draw and resize in one step:
  // Source: video frame, exact region defined by faceBox (no padding)
  // Destination: entire 224x224 canvas
  ctx.drawImage(
    video,
    faceBox.x,
    faceBox.y,
    faceBox.width,
    faceBox.height,
    0,
    0,
    canvas.width,
    canvas.height
  );
  
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

/**
 * Helper to get simple center crop (fallback)
 */
export function detectFaceCenter(
  imageData: ImageData,
  minSize: number = 100
): FaceBox | null {
  const centerX = imageData.width / 2;
  const centerY = imageData.height / 2;
  const size = Math.min(imageData.width, imageData.height) * 0.8;
  
  if (size < minSize) return null;
  
  return {
    x: Math.max(0, centerX - size / 2),
    y: Math.max(0, centerY - size / 2),
    width: size,
    height: size,
  };
}

export function cropFace(
  imageData: ImageData,
  faceBox: FaceBox
): ImageData {
  const canvas = document.createElement('canvas');
  canvas.width = faceBox.width;
  canvas.height = faceBox.height;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('No context');
  
  const srcCanvas = document.createElement('canvas');
  srcCanvas.width = imageData.width;
  srcCanvas.height = imageData.height;
  const srcCtx = srcCanvas.getContext('2d');
  if (!srcCtx) throw new Error('No src context');
  
  srcCtx.putImageData(imageData, 0, 0);
  
  ctx.drawImage(
    srcCanvas,
    faceBox.x,
    faceBox.y,
    faceBox.width,
    faceBox.height,
    0,
    0,
    faceBox.width,
    faceBox.height
  );
  
  return ctx.getImageData(0, 0, faceBox.width, faceBox.height);
}

export function videoFrameToImageData(
  video: HTMLVideoElement
): ImageData {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('No context');
  ctx.drawImage(video, 0, 0);
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}
