# SCUIA Semantic Contrast for Domain-Robust Underwater Image Quality Assessment

## Requirements
Ensure all dependencies are installed by running:

```bash
pip install -r requirements.txt
```

## Testing
1. To evaluate the model, run:

```bash
python predict.py
```

Default mode is folder inference (`predict_image_dir`) and reads images from `test`.

## Usage
### 1) Single image
```bash
python predict.py --eval_type predict_single_image --test_img_path 71.png
```

### 2) Image folder 
```bash
python predict.py --eval_type predict_image_dir --img_dir test
```



