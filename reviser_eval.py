from vis_corrector import Corrector
from types import SimpleNamespace
import argparse
import json
import os, sys
from config import woodpecker_args_dict
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Code for 'Woodpecker: Hallucination Correction for MLLMs Hallucination Correction for MLLMs'.")
    parser.add_argument('--image-path', type=str, help="file path for the text to be corrected.")
    parser.add_argument('--query', type=str, help="text query for MLLM")
    parser.add_argument('--text', type=str, help="text from MLLM to be corrected")
    parser.add_argument('--cache-dir', type=str, help="dir for caching intermediate image",
                        default='./cache_dir')
    
    parser.add_argument('--detector-config', type=str, help="Path to the detector config, \
                        in the form of 'path/to/GroundingDINO_SwinT_OGC.py' ")
    parser.add_argument('--detector-model', type=str, help="Path to the detector checkpoint, \
                        in the form of 'path/to/groundingdino_swint_ogc.pth' ")
    
    parser.add_argument('--api-key', type=str, help="API key for GPT service.")
    parser.add_argument('--api-base', type=str, help="API base link for GPT service.")
    parser.add_argument("--data_path", type=str, default="/home/czr/contrast_decoding_LVLMs/eval_dataset/val2014/", help="data path")
    parser.add_argument(
    "-c",
    "--caption-path",
    type=str,
    required=True,
    help="Path to the generated captions",
    )

    args = parser.parse_args()

    model_args = SimpleNamespace(**woodpecker_args_dict)
    corrector = Corrector(model_args)

    qu = "Please describe this image in detail."
    data_path = args.data_path
    input_captions_path = args.caption_path
    corrected_caption_path = input_captions_path.replace("greedy", "woodpecker")
    loaded_json = []
    with open(input_captions_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            loaded_json.append(json.loads(line))

    prefix = "COCO_val2014_"
    for idx, pair in tqdm.tqdm(enumerate(loaded_json), total=len(loaded_json)):
        img_id = pair['image_id']
        caption = pair['caption']
        img_save = {}
        img_save["image_id"] = img_id
        # pad str(img_id) to have 12 digits with 0, for example, 123 -> 000000000123
        img_id = str(img_id).zfill(12)
        image_path = data_path + prefix + img_id + '.jpg'
        print("image_path", image_path)
        sample = {
        'img_path': image_path,
        'input_desc': caption,
        'query': qu,
        }
        
        corrected_sample = corrector.correct(sample)
        corrected_caption = corrected_sample['output']
        print("input_captions_path", input_captions_path)
        print("original caption: ", caption)
        print("corrected caption: ", corrected_caption)
        # input()

        img_save["caption"] = corrected_caption
        with open(corrected_caption_path, "a", encoding='utf-8') as file:
            json.dump(img_save, file, ensure_ascii=False)
            f.write('\n')