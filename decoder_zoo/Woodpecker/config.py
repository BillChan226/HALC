import sys

sys.path.append("decoder_zoo/Woodpecker")
from openai_setup import api_key, api_base

woodpecker_args_dict = {
    "api_key": api_key,
    "api_base": api_base,
    "val_model_path": "Salesforce/blip2-flan-t5-xxl",
    "qa2c_model_path": "khhuang/zerofec-qa2claim-t5-base",
    "detector_config": "decoder_zoo/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "detector_model_path": "decoder_zoo/GroundingDINO/weights/groundingdino_swint_ogc.pth",
    "cache_dir": "decoder_zoo/HaLC/cache_dir",
}
