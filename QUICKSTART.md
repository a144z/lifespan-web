# Quick Start Guide

## Step 1: Convert Your Model

From the project root directory:

```bash
python web-deploy/convert_to_onnx.py \
  --model_path checkpoints_light/best_mobilenetv3_small_100.pth \
  --model_name mobilenetv3_small_100
```

This creates: `web-deploy/public/models/lifespan_model.onnx`

## Step 2: Install & Run

```bash
cd web-deploy
npm install
npm run dev
```

Visit: http://localhost:3000

## That's it! 🎉

Your web app is ready. Upload a face image to see the prediction.

