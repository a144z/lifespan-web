// Script to verify the ONNX model file exists
const fs = require('fs');
const path = require('path');

const modelPath = path.join(__dirname, '..', 'public', 'models', 'lifespan_model_fastvit.onnx');

if (fs.existsSync(modelPath)) {
  const stats = fs.statSync(modelPath);
  const sizeInMB = (stats.size / (1024 * 1024)).toFixed(2);
  console.log(`✓ Model file found: ${modelPath}`);
  console.log(`  Size: ${sizeInMB} MB`);
  process.exit(0);
} else {
  console.error(`✗ Model file not found: ${modelPath}`);
  console.error(`  Please ensure the model file exists in public/models/`);
  process.exit(1);
}

