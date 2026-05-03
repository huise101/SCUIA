from pathlib import Path


# Configuration file for SCUIA.

PROJECT_ROOT = Path(__file__).resolve().parent

exp_config = {

    # Training entry point used by train.py:
    # 'SCUIA image encoder' trains the SCUIA image encoder.
    # 'SCUIA semantic encoder' trains the SCUIA semantic encoder.
    'run_type': 'SCUIA image encoder',  # 'SCUIA image encoder' or 'SCUIA semantic encoder'
    #'run_type': 'SCUIA semantic encoder',  # 'SCUIA image encoder' or 'SCUIA semantic encoder'

    'database_path': str(PROJECT_ROOT / "Databases"),
    'test_domains': ['UIEB'],
    'predict_test_img_dir': str(PROJECT_ROOT / "UIEB"),
    'predict_test_csv': str(PROJECT_ROOT / "ut" / "UIEB3.CSV"),

    # Training parameters
    'datasets': {
        # Train datasets
        'UIEB': {'train': True},
        #'GT': {'train': True},
    },

    'model': None,  # Model being trained and tested
    'resume_training': False,  # Resume training from existing checkpoint
    'resume_path': str(PROJECT_ROOT),  # Last checkpoint path if resuming training

    'epochs': 40,
    'lr_update': 10,  # Update learning rate after specified no. of epochs
    'test_epoch': 2,  # Validate after these many epochs of training
    'lr_decay': 2.0,

    # Image model arguments
    'batch_size_qacl': 8,  # 9 frames in 1 batch
    'lr_image_model': 1e-4,
    'gt_img_dir': str(PROJECT_ROOT / "Databases" / "GT"),
    'patch_size': 96,
    'device': "cuda",
    'sharpness_param': 0.75,
    'colorfulness_param': 0.8,
    'results_path_image_model': str(r"./Results/Image_Encoder"),

    # Semantic model arguments
    'crop': 'center',
    'crop_size': (224, 224),
    'batch_size_gcl': 66,
    'tau': 32,  # temperature parameter
    'lr_semantic_model': 1e-4,
    'results_path_semantic_model': str(r"./Results/Semantic_Encoder"),

    # Score fusion arguments
    'score_fusion_image_weight': 0.9995,
    'score_calibration_min': 0.42,
    'score_calibration_max': 0.52,
}
