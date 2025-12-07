// Script to verify the ONNX model file exists and is accessible
const fs = require('fs');
const path = require('path');

const modelPath = path.join(__dirname, '..', 'public', 'models', 'lifespan_model_fastvit.onnx');

console.log('Verifying ONNX model file...');
console.log('Expected location:', modelPath);

if (fs.existsSync(modelPath)) {
  const stats = fs.statSync(modelPath);
  const fileSizeMB = (stats.size / (1024 * 1024)).toFixed(2);
  console.log(`✓ Model file found!`);
  console.log(`  Size: ${fileSizeMB} MB (${stats.size} bytes)`);
  console.log(`  Last modified: ${stats.mtime}`);
  
  // Check if file is readable
  try {
    fs.accessSync(modelPath, fs.constants.R_OK);
    console.log(`✓ File is readable`);
  } catch (error) {
    console.error(`✗ File is not readable:`, error.message);
    process.exit(1);
  }
  
  console.log('\n✓ Model file verification passed!');
  console.log('  If you still see 404 errors:');
  console.log('  1. Stop the dev server (Ctrl+C)');
  console.log('  2. Delete .next folder: rm -rf .next (or rmdir /s .next on Windows)');
  console.log('  3. Restart: npm run dev');
} else {
  console.error(`✗ Model file NOT FOUND at: ${modelPath}`);
  console.error('\nPlease ensure:');
  console.error('  1. The file exists at: public/models/lifespan_model_fastvit.onnx');
  console.error('  2. The file name matches exactly (case-sensitive)');
  console.error('  3. The file was not deleted or moved');
  process.exit(1);
}

