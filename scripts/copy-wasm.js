// Script to copy WASM files from node_modules to public directory
const fs = require('fs');
const path = require('path');

const sourceDir = path.join(__dirname, '..', 'node_modules', 'onnxruntime-web', 'dist');
const destDir = path.join(__dirname, '..', 'public', 'wasm');

// Create destination directory if it doesn't exist
if (!fs.existsSync(destDir)) {
  fs.mkdirSync(destDir, { recursive: true });
}

// Copy WASM files
const wasmFiles = [
  'ort-wasm.wasm',
  'ort-wasm-simd.wasm',
  'ort-wasm-threaded.wasm',
  'ort-wasm-simd-threaded.wasm',
];

let copiedCount = 0;
wasmFiles.forEach((file) => {
  const sourcePath = path.join(sourceDir, file);
  const destPath = path.join(destDir, file);
  
  if (fs.existsSync(sourcePath)) {
    try {
      fs.copyFileSync(sourcePath, destPath);
      console.log(`✓ Copied ${file}`);
      copiedCount++;
    } catch (error) {
      console.warn(`⚠ Failed to copy ${file}:`, error.message);
    }
  } else {
    console.warn(`⚠ ${file} not found in ${sourceDir}`);
  }
});

if (copiedCount > 0) {
  console.log(`\n✓ Copied ${copiedCount} WASM file(s) to public/wasm/`);
} else {
  console.log('\n⚠ No WASM files were copied. Using CDN fallback.');
}

