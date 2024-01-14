import os
import json
import argparse
import sys
import copy

sys.path.append("../HALC")
import numpy as np
from tqdm import tqdm

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

import json

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from collections import defaultdict

import argparse

# Set the directory where the chair.json files are located
# directory = './paper_result/32_tokens/minigpt4/'

parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")

parser.add_argument(
    "-c",
    "--caption-path",
    type=str,
    required=True,
    help="Path to the generated captions",
)

args = parser.parse_known_args()[0]

directory = args.caption_path


# Assuming this script is placed in the same directory as the JSON files
directory_path = directory
output_directory_path = directory
# List all files in the directory
files = os.listdir(directory_path)

# Filter out files that do not end with '_generated_captions.json'
caption_files = [file for file in files if file.endswith("_generated_captions.jsonl")]

# loaded_json = json.load(open(generated_captions_path))
caption_file_path = (
    "/media/zhuokai/SN850X_4TB/Data/coco/annotations_all/captions_val2014.json"
)
annotation_file_path = (
    "/media/zhuokai/SN850X_4TB/Data/coco/annotations_all/captions_val2014.json"
)
# with open(args.data_path + '../annotations_trainval2014/annotations/instances_val2014.json', 'r') as f:
with open(annotation_file_path, "r") as f:
    lines = f.readlines()
coco_anns = json.loads(lines[0])

coco = COCO(caption_file_path)


for file_name in caption_files:
    # Construct the full file path
    file_path = os.path.join(directory_path, file_name)

    # Process the file (you would insert your processing code here)
    # For example, load the JSON, perform operations, and then save the output

    print("file_path: ", file_path)

    loaded_json = []
    with open(file_path, "r") as f:
        try:
            lines = f.readlines()
            for line in lines:
                loaded_json.append(json.loads(line))
        except:
            continue

    # eliminate the items in loaded_json with the same key:
    for i in range(len(loaded_json)):
        for j in range(i + 1, len(loaded_json)):
            if loaded_json[i]["image_id"] == loaded_json[j]["image_id"]:
                loaded_json.pop(j)
                break

    # # save loaded json

    # generated_captions_path = file_name.replace('_generated_captions.json', '_generated_captions_new.json')
    # output_file_path = os.path.join(output_directory_path, generated_captions_path)
    # with open(output_file_path, "a") as f:
    #     json.dump(img_save, f)
    #     f.write('\n')

    print("loaded_json: ", len(loaded_json))
    # construct output file as input to CHAIR evaluation
    # output format follows https://github.com/ruotianluo/self-critical.pytorch
    formulated_output_dict = {}
    # overall result
    all_overall_scores = defaultdict(list)
    # imgToEval per image result
    img_to_eval_dict = {}
    good = copy.deepcopy(loaded_json)
    loaded_json = []
    loaded_json = good
    # to save memory, load 100 captions at a time
    for start_idx in tqdm(
        range(0, len(loaded_json), 100), desc="Generating CHAIR Input"
    ):
        # define the current iteration end index
        end_idx = min(start_idx + 100, len(loaded_json))
        coco_res = coco.loadRes(
            loaded_json[start_idx:end_idx],
        )

        coco_eval = COCOEvalCap(coco, coco_res)
        coco_eval.params["image_id"] = coco_res.getImgIds()
        coco_eval.evaluate()

        # keep track of the overall scores
        for metric, score in coco_eval.eval.items():
            all_overall_scores[metric].append(score)

        # imgToEval per image result
        for i, cur_img_id in enumerate(coco_res.getImgIds()):
            cur_eval_dict = coco_eval.evalImgs[i]
            # add caption to the eval dict
            cur_eval_dict["caption"] = coco_res.imgToAnns[cur_img_id][0]["caption"]
            img_to_eval_dict[cur_img_id] = cur_eval_dict

    # overall result
    overall_dict = {}
    for metric, score in all_overall_scores.items():
        overall_dict[metric] = np.mean(score)
    formulated_output_dict["overall"] = overall_dict
    formulated_output_dict["imgToEval"] = img_to_eval_dict

    # Construct the output file name by replacing the ending
    output_file_name = file_name.replace("_generated_captions.json", "_chair.json")
    output_file_path = os.path.join(output_directory_path, output_file_name)

    # Save the processed data to the new file
    with open(output_file_path, "w") as f_out:
        json.dump(
            formulated_output_dict, f_out
        )  # Assuming processed_data is the result of your processing

    print("output_file_path: ", output_file_path)
