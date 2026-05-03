# SCUIA Semantic Contrast for Domain-Robust Underwater Image Quality Assessment

## Requirements
Install dependencies:

```bash
pip install -r requirements.txt
```

## Model Weights
- [Image encoder](https://drive.google.com/file/d/1uV6JDiC5-4z1GcnRIDD3htujc5Tp-tkQ/view?usp=sharing)
- [Semantic encoder](https://drive.google.com/file/d/1JuuKoCZB1cjercldaxh7B9eUzeLKq2S1/view?usp=sharing)
- [GT cache (`gt.hdf5`)](https://drive.google.com/file/d/1FrUCZOtxodFax5rxeXH012imERc2FszO/view?usp=sharing)

### Download with `gdown`
```bash
pip install gdown
gdown --id 1uV6JDiC5-4z1GcnRIDD3htujc5Tp-tkQ -O checkPoints/scuia_image_encoder_model.tar
gdown --id 1JuuKoCZB1cjercldaxh7B9eUzeLKq2S1 -O checkPoints/scuia_semantic_encoder_model.pth
gdown --id 1FrUCZOtxodFax5rxeXH012imERc2FszO -O gt.hdf5
```

## Expected File Layout
```text
SCUIA/
  checkPoints/
    scuia_image_encoder_model.tar
    scuia_semantic_encoder_model.pth
  gt.hdf5
  predict.py
```

## Testing
Run:

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

