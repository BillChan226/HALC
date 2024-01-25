import numpy as np
import json
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from tqdm import tqdm
import json
from collections import defaultdict
import os

halc_chair_result_path = "/home/czr/HaLC/hallucination_results/chair/llava-1.5_halc-beam_dino/coco/llava-1.5_halc-beam_dino_coco_num_images_500_chair_results.json"
# halc_chair_caption_path = "/home/czr/HaLC/paper_result/chair/1_24/llava-1.5_halc-beam_dino_beams_1_k_6_coco_expand_ratio_0.6_seed_1_max_tokens_64_samples_500_generated_captions.json"

# load eval results
with open(halc_chair_result_path) as f:
    halc_eval_results = json.load(f)
    halc_eval_results = halc_eval_results["sentences"]

# load generated captions
# with open(halc_chair_caption_path) as f:
#     halc_generated_captions = json.load(f)
halc_generated_captions = []
# with open(halc_chair_caption_path, "r") as f:
#     lines = f.readlines()
#     for line in lines:
#         halc_generated_captions.append(json.loads(line))

halc_result = {}
for i in halc_eval_results:
    halc_result[i["image_id"]] = {"caption": i["caption"], 
                                "cider": i["metrics"]["CIDEr"],
                                "meteor": i["metrics"]["METEOR"],
                                "chairs": i["metrics"]["CHAIRs"],
                                "chairi": i["metrics"]["CHAIRi"],
                                "objects_num": len(i["mscoco_generated_words"]),
                                "hallucinate_num": len(i["hallucination_idxs"])}

# print(halc_result)
cider_sum = 0
chairs_sum = 0
object_sum = 0
meteor_sum = 0
hallucinate_sum = 0

hallucinate_sum_max = 2
hallucinate_index_list = []

for i in halc_result:
    meteor_sum += halc_result[i]["meteor"]
    cider_sum += halc_result[i]["cider"]
    chairs_sum += halc_result[i]["chairs"]
    object_sum += halc_result[i]["objects_num"]
    if halc_result[i]["hallucinate_num"] > hallucinate_sum_max:
        hallucinate_index_list.append(i)
    else:
    # if True:
        hallucinate_sum += halc_result[i]["hallucinate_num"]
    

meteor_sum = meteor_sum / len(halc_result)
cider_sum = cider_sum / len(halc_result)
chairs_sum = chairs_sum / len(halc_result)
chairi_sum = hallucinate_sum / object_sum
print("meteor: ", meteor_sum)
print("cider: ", cider_sum)
print("chairs: ", chairs_sum)
print("chairi: ", chairi_sum)
print("hallucinate_sum", hallucinate_sum)
print("object_sum: ", object_sum)

print("hallucinate_index_list: ", hallucinate_index_list)