# Fixing ONNX DLL Error on Windows

## Error Message
```
ImportError: DLL load failed while importing onnx_cpp2py_export
```

## Quick Fix (Try These in Order)

### Solution 1: Reinstall ONNX
```bash
pip uninstall onnx
pip install onnx --no-cache-dir
```

### Solution 2: Install Visual C++ Redistributables
1. Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Install it
3. **Restart your terminal/IDE**
4. Try again

### Solution 3: Use Conda (Recommended)
```bash
# Create fresh environment
conda create -n onnx_env python=3.10
conda activate onnx_env

# Install everything
pip install torch torchvision timm onnx

# Then run conversion
python web-deploy/convert_to_onnx.py --model_path <path>
```

### Solution 4: Try Different ONNX Version
```bash
pip uninstall onnx
pip install onnx==1.15.0
```

### Solution 5: Use Alternative Script
If ONNX still doesn't work, the conversion might still succeed:
```bash
python web-deploy/convert_to_onnx_alt.py --model_path <path> --model_name <name>
```

## Why This Happens

ONNX uses C++ extensions that require:
- Visual C++ Redistributables (Windows)
- Compatible Python version
- Proper DLL loading

## Verification

After fixing, test with:
```bash
python -c "import onnx; print('OK:', onnx.__version__)"
```

If this works, the conversion should work too.

