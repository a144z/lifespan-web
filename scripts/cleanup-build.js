// Post-build script to remove Node.js-specific ONNX files
const fs = require('fs');
const path = require('path');

function deleteFilesRecursive(dir, pattern) {
  if (!fs.existsSync(dir)) {
    return;
  }
  
  const files = fs.readdirSync(dir);
  
  for (const file of files) {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);
    
    if (stat.isDirectory()) {
      deleteFilesRecursive(filePath, pattern);
    } else if (pattern.test(file)) {
      try {
        fs.unlinkSync(filePath);
        console.log(`Deleted: ${filePath}`);
      } catch (err) {
        console.error(`Error deleting ${filePath}:`, err);
      }
    }
  }
}

// Clean up .next directory
const nextDir = path.join(__dirname, '..', '.next');
if (fs.existsSync(nextDir)) {
  deleteFilesRecursive(nextDir, /ort\.node\.min\.mjs/);
  console.log('Cleanup completed');
} else {
  console.log('.next directory not found, skipping cleanup');
}

