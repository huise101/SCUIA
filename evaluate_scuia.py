import warnings
import os


warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from CrossDomainDegradationGuidance import *
from semantic_encoder_eval import *
from networks import *
from configs import exp_config
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import sys
import torch
import traceback


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = r"test"
DEFAULT_SINGLE_IMAGE = r'71.png'

DEFAULT_GT_DIR = PROJECT_ROOT / 'GT'

EVAL_TYPE_PREDICT_SINGLE_IMAGE = 'predict_single_image'
EVAL_TYPE_PREDICT_IMAGE_DIR = 'predict_image_dir'


DEFAULT_EVAL_CONFIG = {
    # Use 'predict_single_image' for one-image prediction.
    # Use 'predict_image_dir' for folder batch prediction without CSV.
    'eval_type': EVAL_TYPE_PREDICT_IMAGE_DIR,
    'img_dir': DEFAULT_IMAGE_DIR,
    'test_img_path': DEFAULT_SINGLE_IMAGE,
}


def _as_scalar_score(score):
    if isinstance(score, torch.Tensor):
        return score.detach().cpu().reshape(-1)[0].item()
    if isinstance(score, np.ndarray):
        return float(score.reshape(-1)[0])
    if isinstance(score, (list, tuple)):
        return _as_scalar_score(score[0])
    return float(score)


class ScoreFusionMLP(torch.nn.Module):
    def __init__(self, image_score_weight):
        super().__init__()
        semantic_score_weight = 1.0 - image_score_weight
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1),
        )
        self._init_fixed_fusion(image_score_weight, semantic_score_weight)
        for param in self.parameters():
            param.requires_grad_(False)

    def _init_fixed_fusion(self, image_score_weight, semantic_score_weight):
        with torch.no_grad():
            self.net[0].weight.zero_()
            self.net[0].bias.zero_()
            self.net[0].weight[0, 0] = 1.0
            self.net[0].weight[1, 1] = 1.0

            self.net[2].weight.zero_()
            self.net[2].bias.zero_()
            self.net[2].weight[0, 0] = image_score_weight
            self.net[2].weight[0, 1] = semantic_score_weight

    def forward(self, image_score, semantic_score):
        image_score = torch.as_tensor(image_score, dtype=torch.float32).reshape(-1, 1)
        semantic_score = torch.as_tensor(semantic_score, dtype=torch.float32).reshape(-1, 1)
        scores = torch.cat([image_score, semantic_score], dim=1)
        return self.net(scores).squeeze(-1)


def calibrate_quality_score(score):
    score_min = exp_config['score_calibration_min']
    score_max = exp_config['score_calibration_max']
    if score_max <= score_min:
        raise ValueError('score_calibration_max must be greater than score_calibration_min')
    calibrated = (score - score_min) / (score_max - score_min)
    return np.clip(calibrated, 0.0, 1.0)


def parse_option():
    parser = argparse.ArgumentParser('arguments for evaluation')

    parser.add_argument('--device', type=str, default='cuda:0', help='Device (cpu/cuda)')
    parser.add_argument(
        '--image_encoder_weights_path',
        type=str,
        default=str(PROJECT_ROOT / 'checkPoints' / 'scuia_image_encoder_model.tar'),
        help='Image encoder model weights path',
    )
    parser.add_argument(
        '--semantic_encoder_weights_path',
        type=str,
        default=str(PROJECT_ROOT / 'checkPoints' / 'scuia_semantic_encoder_model.pth'),
        help='Semantic encoder model weights path',
    )
    parser.add_argument(
        '--eval_type',
        type=str,
        default=DEFAULT_EVAL_CONFIG['eval_type'],
        help='Evaluation modes (predict_single_image/predict_image_dir)',
    )
    parser.add_argument(
        '--img_dir',
        type=str,
        default=str(DEFAULT_EVAL_CONFIG['img_dir']),
        help='Image directory for above chosen dataset',
    )
    parser.add_argument(
        '--test_img_path',
        type=str,
        default=str(DEFAULT_EVAL_CONFIG['test_img_path']),
        help='Test image path for predict_single_image evaluation',
    )
    parser.add_argument(
        '--gt_img_dir',
        type=str,
        default=str(DEFAULT_GT_DIR),
        help='Image directory for GT images.',
    )
    parser.add_argument('--patch_size', default=96, type=int, help='Patch size for GT patches')
    parser.add_argument(
        '--sharpness_param',
        default=0.75,
        type=float,
        help='Sharpness parameter for selecting GT patches',
    )
    parser.add_argument(
        '--colorfulness_param',
        default=0.8,
        type=float,
        help='Colorfulness parameter for selecting GT patches',
    )
    args = parser.parse_args()
    if args.eval_type not in {EVAL_TYPE_PREDICT_SINGLE_IMAGE, EVAL_TYPE_PREDICT_IMAGE_DIR}:
        raise ValueError(
            f"Unsupported eval_type '{args.eval_type}'. Use {EVAL_TYPE_PREDICT_SINGLE_IMAGE} or {EVAL_TYPE_PREDICT_IMAGE_DIR}."
        )
    if not str(args.device).lower().startswith('cuda'):
        raise ValueError("SCUIA evaluation is configured for GPU-only execution. Please set --device to a CUDA device.")
    return args


class PredictionEvaluation:
    def __init__(self, args, image_encoder, semantic_encoder):
        self.image_encoder = image_encoder
        self.semantic_encoder = semantic_encoder
        self.args = args
        self.score_fusion = ScoreFusionMLP(exp_config['score_fusion_image_weight'])

    def predict_single_image(self):
        test_image_path = self.args.test_img_path
        print(f"[INFO] Mode: single image")
        print(f"[INFO] Input image: {test_image_path}")
        print("[INFO] Building image reference module (F_D)...")

        fd_module = build_degradation_guidance_module(self.image_encoder, self.args)
        print("[INFO] Building semantic inference context...")
        semantic_context = build_semantic_inference_context(self.semantic_encoder)
        print("[INFO] Running prediction...")
        score_image = compute_degradation_guidance_single_image(
            self.image_encoder, test_image_path, self.args, fd_module=fd_module, clear_cache=True
        )
        score_semantic = compute_semantic_score_single_image(
            self.semantic_encoder, test_image_path, inference_context=semantic_context
        )

        score_image = _as_scalar_score(score_image)
        score_semantic = _as_scalar_score(score_semantic)
        score_raw = self.score_fusion([score_image], [score_semantic]).item()
        combined = float(calibrate_quality_score(score_raw))
        file_name = Path(test_image_path).name
        output_csv = 'predict_eval_results_single_image.csv'
        pd.DataFrame([{'file_name': file_name, 'combined': combined}]).to_csv(
            output_csv, mode='a', index=False, header=not os.path.exists(output_csv)
        )
        print(f"{file_name},{combined}")
        print(f"[INFO] Appended to CSV: {output_csv}")

    def predict_image_dir(self):
        img_dir = Path(self.args.img_dir)
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: '{img_dir}'")

        output_csv = f'predict_eval_results_{img_dir.name}.csv'
        valid_suffixes = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
        image_files = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_suffixes])
        if not image_files:
            raise ValueError(f"No supported image files found in '{img_dir}'")
        print(f"[INFO] Mode: image directory")
        print(f"[INFO] Input directory: {img_dir}")
        print(f"[INFO] Images found: {len(image_files)}")
        print(f"[INFO] Output CSV: {output_csv}")
        print("[INFO] Building image reference module (F_D)...")

        fd_module = build_degradation_guidance_module(self.image_encoder, self.args)
        print("[INFO] Building semantic inference context...")
        semantic_context = build_semantic_inference_context(self.semantic_encoder)
        print("[INFO] Starting batch prediction...")
        combined_scores = []

        for idx, image_path in enumerate(image_files, start=1):
            print(f"[INFO] Processing {idx}/{len(image_files)}: {image_path.name}")
            score_image = compute_degradation_guidance_single_image(
                self.image_encoder, str(image_path), self.args, fd_module=fd_module, clear_cache=False
            )
            score_semantic = compute_semantic_score_single_image(
                self.semantic_encoder, str(image_path), inference_context=semantic_context
            )
            score_image = _as_scalar_score(score_image)
            score_semantic = _as_scalar_score(score_semantic)
            score_raw = self.score_fusion([score_image], [score_semantic]).item()
            combined = float(calibrate_quality_score(score_raw))
            combined_scores.append(combined)

            pd.DataFrame([{'file_name': image_path.name, 'combined': combined}]).to_csv(
                output_csv, mode='a', index=False, header=not os.path.exists(output_csv)
            )
            print(f"{image_path.name},{combined}")

        dataset_average = float(np.mean(combined_scores))
        pd.DataFrame([{'file_name': '__average__', 'combined': dataset_average}]).to_csv(
            output_csv, mode='a', index=False, header=not os.path.exists(output_csv)
        )
        print(f"__average__,{dataset_average}")
        print(f"[INFO] Appended to CSV: {output_csv}")

def eval_mode(model):
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    return model


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model_checkpoint = checkpoint['model']
        if isinstance(model_checkpoint, dict) and 'state_dict' in model_checkpoint:
            return model_checkpoint['state_dict']
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    return checkpoint


def _load_image_encoder_state(model, checkpoint):
    state_dict = _extract_state_dict(checkpoint)
    model_state = model.state_dict()
    compatible_state = {}

    for key, value in state_dict.items():
        candidate_key = key
        if key.startswith(('am.', 'ff.', 'dy.')):
            candidate_key = f'addb.{key}'

        candidate_key = candidate_key.replace('.sobel_x1.', '.sobel_x.')
        candidate_key = candidate_key.replace('.sobel_y1.', '.sobel_y.')

        if candidate_key in model_state and model_state[candidate_key].shape == value.shape:
            compatible_state[candidate_key] = value

    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    if missing:
        raise RuntimeError(f'ImageEncoder checkpoint is missing compatible keys: {missing}')
    if unexpected:
        raise RuntimeError(f'Unexpected ImageEncoder keys after filtering: {unexpected}')


def load_model(model_weights_path, network_type):
    if network_type == 'image':
        model = ImageEncoder(encoder='resnet18', head='mlp').to("cuda")
        load_dict = torch.load(model_weights_path, map_location="cuda")
        _load_image_encoder_state(model, load_dict)
    elif network_type == 'semantic':
        model = SemanticEncoder().to("cuda")
        load_dict = torch.load(model_weights_path, map_location="cuda")
        model.clip_model.visual.load_state_dict(load_dict, strict=False)
    else:
        raise ValueError(f'Unsupported network_type: {network_type}')

    return model


def main():
    args = parse_option()
    print("[INFO] Loading image encoder...")

    image_encoder = load_model(model_weights_path=args.image_encoder_weights_path, network_type='image')
    image_encoder = eval_mode(model=image_encoder)
    print("[INFO] Loading semantic encoder...")

    semantic_encoder = load_model(model_weights_path=args.semantic_encoder_weights_path, network_type='semantic')
    semantic_encoder = eval_mode(model=semantic_encoder)
    print("[INFO] Models loaded. Starting evaluation...")

    predictor = PredictionEvaluation(args, image_encoder, semantic_encoder)
    if args.eval_type == EVAL_TYPE_PREDICT_SINGLE_IMAGE:
        predictor.predict_single_image()
    elif args.eval_type == EVAL_TYPE_PREDICT_IMAGE_DIR:
        predictor.predict_image_dir()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()
        sys.exit(1)
