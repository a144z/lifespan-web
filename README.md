# Remaining Lifespan Detector

A modern web application that uses artificial intelligence to estimate remaining lifespan from facial images. The application runs entirely in your browser, ensuring complete privacy and security.

## Overview

The Remaining Lifespan Detector is an interactive web tool that analyzes facial features to provide an estimate of how many years a person might have remaining. The application features a beautiful, user-friendly interface and supports both photo upload and real-time camera detection.

## Features

### 🎯 Core Functionality
- **Face Detection**: Advanced face detection using MediaPipe technology
- **Real-time Processing**: Live camera mode with instant predictions
- **Photo Upload**: Upload and analyze static images
- **Visual Feedback**: Green bounding box shows detected face area
- **Instant Results**: Predictions update in real-time as you move

### 🔒 Privacy & Security
- **100% Browser-Based**: All processing happens locally in your browser
- **No Data Transmission**: Your photos never leave your device
- **No Server Required**: Complete client-side operation
- **Zero Tracking**: No analytics or user tracking

### 🎨 User Experience
- **Modern Design**: Clean, intuitive interface with smooth animations
- **Responsive Layout**: Works seamlessly on desktop, tablet, and mobile devices
- **Real-time Feedback**: Live FPS counter and visual indicators
- **Error Handling**: Clear error messages and helpful guidance

### ⚡ Performance
- **Fast Inference**: Optimized for quick predictions (~10 FPS in camera mode)
- **Lightweight**: Efficient model architecture for smooth performance
- **Cross-Platform**: Works on all modern browsers with WebAssembly support

## How It Works

### Upload Mode
1. Click the "Upload Photo" button
2. Select a face image from your device
3. The application automatically detects the face
4. View the predicted remaining lifespan in years

### Live Camera Mode
1. Click the "Live Camera" button
2. Grant camera permissions when prompted
3. Position your face in the center of the frame
4. Watch the green bounding box track your face
5. See predictions update in real-time as you move

## Understanding the Results

### Prediction Range
- **Maximum Prediction**: Approximately 50 years
- **High Predictions (40-50 years)**: Indicates potential for a long life based on patterns in the training data
- **Lower Predictions**: Based on patterns observed in the dataset

### Important Notes
- Predictions are based on facial features and patterns learned from historical data
- Results are estimates, not medical diagnoses
- The model was trained on adult faces, not children or babies
- Maximum prediction is limited by the training dataset

## Technical Details

### Model Architecture
- **Type**: Vision Transformer (FastViT) based model
- **Input Size**: 224x224 pixels
- **Output**: Single value representing remaining lifespan in years
- **Format**: ONNX (Open Neural Network Exchange) for browser compatibility

### Face Detection
- **Technology**: MediaPipe Face Detection
- **Method**: Real-time face detection with bounding box visualization
- **Accuracy**: Precise face boundary detection
- **Performance**: Optimized for real-time processing

### Browser Requirements
- Modern browser with WebAssembly support
- Camera access (for live mode)
- JavaScript enabled
- Recommended: Chrome, Firefox, Safari, or Edge (latest versions)

## Getting Started

### Installation

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Add Model File**
   - Place `lifespan_model_fastvit.onnx` in the `public/models/` directory
   - Ensure the file is named exactly: `lifespan_model_fastvit.onnx`

3. **Start Development Server**
   ```bash
   npm run dev
   ```

4. **Open in Browser**
   - Navigate to `http://localhost:3000`
   - Allow camera permissions if using live mode

### Building for Production

```bash
npm run build
npm start
```

## Project Structure

```
web-deploy/
├── app/
│   ├── layout.tsx          # Application layout and metadata
│   ├── page.tsx            # Main application page with UI
│   └── globals.css         # Global styles and theming
├── lib/
│   ├── onnxModel.ts        # ONNX model loading and inference
│   └── faceDetection.ts   # Face detection using MediaPipe
├── public/
│   └── models/
│       └── lifespan_model_fastvit.onnx  # AI model file
├── package.json            # Dependencies and scripts
├── next.config.js         # Next.js configuration
├── tsconfig.json          # TypeScript configuration
└── tailwind.config.js     # Tailwind CSS configuration
```

## Usage Tips

### For Best Results
- **Good Lighting**: Ensure your face is well-lit
- **Face Position**: Center your face in the frame
- **Clear View**: Remove glasses or obstructions if possible
- **Still Position**: Hold still for a moment for accurate detection

### Camera Mode
- Position yourself 1-2 feet from the camera
- Ensure your entire face is visible
- The green box indicates successful face detection
- Predictions update automatically when a face is detected

### Upload Mode
- Use clear, front-facing photos
- Ensure the face is clearly visible
- Higher resolution images work better
- Single person photos are recommended

## Troubleshooting

### Model Not Loading
- Verify `lifespan_model_fastvit.onnx` exists in `public/models/`
- Check browser console for error messages
- Ensure the file name matches exactly
- Try refreshing the page

### Camera Not Working
- Grant camera permissions when prompted
- Check browser settings for camera access
- Ensure no other application is using the camera
- Try a different browser if issues persist

### Face Not Detected
- Ensure good lighting conditions
- Position face in the center of the frame
- Remove obstructions (masks, hands, etc.)
- Try moving closer or further from the camera

### Slow Performance
- Close other browser tabs to free up resources
- Use a modern browser with latest updates
- Ensure sufficient device memory available
- Camera mode may be slower on older devices

## Browser Compatibility

### Fully Supported
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

### Features by Browser
- **WebAssembly**: Required for model inference (all modern browsers)
- **MediaStream API**: Required for camera mode (all modern browsers)
- **Canvas API**: Required for face visualization (all modern browsers)

## Privacy & Data

### What We Don't Collect
- No images are stored
- No predictions are logged
- No user data is collected
- No analytics or tracking

### What Happens Locally
- Images are processed in browser memory
- Model runs entirely on your device
- All data is cleared when you close the page
- No network requests for predictions

## Limitations & Disclaimers

### Accuracy Disclaimer
- **Not Medical Advice**: This tool is for entertainment and educational purposes only
- **Not Scientifically Validated**: Results should not be used for health decisions
- **Pattern-Based**: Predictions are based on patterns in training data, not medical science

### Model Limitations
- **Age Range**: Not trained on babies or children - results may not be meaningful for young people
- **Maximum Value**: Predictions are capped around 50 years based on training data
- **Dataset Bias**: Results reflect patterns in the training dataset, which may have limitations

### Technical Limitations
- Requires modern browser with WebAssembly
- Camera access required for live mode
- Performance varies by device capabilities
- Internet connection not required (after initial load)

## Support

For issues, questions, or feedback:
- Check browser console for error messages
- Ensure all requirements are met
- Verify model file is correctly placed
- Try different browsers or devices

## License

This web application is part of the Remaining Lifespan Prediction project. Please refer to the main project license for usage terms.

---

**Remember**: This tool is for entertainment purposes only. The predictions are estimates based on facial patterns and should never be used for medical or health-related decisions.
"# lifespan-web" 
