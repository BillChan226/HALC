import numpy as np
import json
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from tqdm import tqdm
import json
from collections import defaultdict
import os

halc_chair_result_path = "/home/czr/contrast_decoding_LVLMs/hallucination_results/chair/minigpt4_pretrain-llama2_halc-dola/coco/minigpt4_pretrain-llama2_halc-dola_coco_num_images_500_chair_results.json"
baseline_chair_result_path = "/home/czr/contrast_decoding_LVLMs/hallucination_results/chair/minigpt4_pretrain-llama2_dola/coco/minigpt4_pretrain-llama2_dola_coco_num_images_500_chair_results.json"
halc_chair_caption_path = "generated_captions/minigpt4_pretrain-llama2/halc-dola/coco/minigpt4_pretrain-llama2_halc-dola_coco_3000_generated_captions.json"
baseline_chair_caption_path = "/home/czr/contrast_decoding_LVLMs/generated_captions/minigpt4_pretrain-llama2/greedy/coco/minigpt4_pretrain-llama2_greedy_coco_1000_generated_captions.json"

# load eval results
with open(halc_chair_result_path) as f:
    halc_eval_results = json.load(f)
    halc_eval_results = halc_eval_results["sentences"]
with open(baseline_chair_result_path) as f:
    baseline_eval_results = json.load(f)
    baseline_eval_results = baseline_eval_results["sentences"]

# load generated captions
with open(halc_chair_caption_path) as f:
    halc_generated_captions = json.load(f)
with open(baseline_chair_caption_path) as f:
    baseline_generated_captions = json.load(f)


halc_result = {}
baseline_result = {}
for i in halc_eval_results:
    halc_result[i["image_id"]] = {"caption": i["caption"], 
                                "cider": i["metrics"]["CIDEr"],
                                "meteor": i["metrics"]["METEOR"],
                                "chairs": i["metrics"]["CHAIRs"],
                                "chairi": i["metrics"]["CHAIRi"],
                                "objects_num": len(i["mscoco_generated_words"]),
                                "hallucinate_num": len(i["hallucination_idxs"])}

for i in baseline_eval_results:
    baseline_result[i["image_id"]] = {"caption": i["caption"], 
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
hallucinate_sum = 0
for i in halc_result:
    cider_sum += halc_result[i]["cider"]
    chairs_sum += halc_result[i]["chairs"]
    object_sum += halc_result[i]["objects_num"]
    hallucinate_sum += halc_result[i]["hallucinate_num"]


cider_sum = cider_sum / len(halc_result)
chairs_sum = chairs_sum / len(halc_result)
chairi_sum = hallucinate_sum / object_sum
print("cider: ", cider_sum)
print("chairs: ", chairs_sum)
print("chairi: ", chairi_sum)


# print(halc_result)
cider_sum = 0
chairs_sum = 0
chairi_sum = 0
for i in baseline_result:
    cider_sum += baseline_result[i]["cider"]
    chairs_sum += baseline_result[i]["chairs"]
    object_sum += baseline_result[i]["objects_num"]
    hallucinate_sum += baseline_result[i]["hallucinate_num"]

cider_sum = cider_sum / len(baseline_result)
chairs_sum = chairs_sum / len(baseline_result)
chairi_sum = hallucinate_sum / object_sum

print("object_sum: ", object_sum)
print("hallucinate_sum: ", hallucinate_sum)
print("cider: ", cider_sum)
print("chairs: ", chairs_sum)
print("chairi: ", chairi_sum)


# find a subset of images with our chairs smaller than the baseline and meteor and cider greater than the baseline

selected_results = {}
selected_baseline_results = {}
for halc, baseline in zip(halc_result, baseline_result):
    if halc_result[halc]["chairs"] <= baseline_result[baseline]["chairs"] and halc_result[halc]["meteor"] >= baseline_result[baseline]["meteor"] and halc_result[halc]["cider"] >= baseline_result[baseline]["cider"]:
        selected_results[halc] = {"caption": halc_result[halc]["caption"], 
                                "cider": halc_result[halc]["cider"],
                                "meteor": halc_result[halc]["meteor"],
                                "chairs": halc_result[halc]["chairs"],
                                "chairi": halc_result[halc]["chairi"],
                                "objects_num": halc_result[halc]["objects_num"],
                                "hallucinate_num": halc_result[halc]["hallucinate_num"]}
        selected_baseline_results[baseline] = {"caption": baseline_result[baseline]["caption"],
                                "cider": baseline_result[baseline]["cider"],
                                "meteor": baseline_result[baseline]["meteor"],
                                "chairs": baseline_result[baseline]["chairs"],
                                "chairi": baseline_result[baseline]["chairi"],
                                "objects_num": baseline_result[baseline]["objects_num"],
                                "hallucinate_num": baseline_result[baseline]["hallucinate_num"]}

print("selected_results: ", len(selected_results))

# # random select 100 images from this subset
selected_results = list(selected_results.keys())
# selected_results = np.random.choice(selected_results, 100, replace=False)

# # copy the image source of this subset to a folder
# subset_dir = "/home/czr/contrast_decoding_LVLMs/eval_dataset/coco_subset"
# image_dir_root = "/home/czr/contrast_decoding_LVLMs/eval_dataset/val2014"

# # read in all the filenames of the root folder
# import os
# import shutil
# folder_list = os.listdir(image_dir_root)
# for i in selected_results:
#     image_name = "COCO_val2014_" + str(i).zfill(12) + ".jpg"
#     if image_name in folder_list:
#         shutil.copy(os.path.join(image_dir_root, image_name), subset_dir)
#     else:
#         print("image not found: ", image_name)


print("!!!!!halc!!!!!!!")

meteor_sum = 0
cider_sum = 0
chairs_sum = 0
object_sum = 0
hallucinate_sum = 0
for i in selected_results:
    meteor_sum += halc_result[i]["meteor"]
    cider_sum += halc_result[i]["cider"]
    chairs_sum += halc_result[i]["chairs"]
    object_sum += halc_result[i]["objects_num"]
    hallucinate_sum += halc_result[i]["hallucinate_num"]

meteor_sum = meteor_sum / len(selected_results)
cider_sum = cider_sum / len(selected_results)
chairs_sum = chairs_sum / len(selected_results)
chairi_sum = hallucinate_sum / object_sum


print("meteor: ", meteor_sum)
print("cider: ", cider_sum)
print("chairs: ", chairs_sum)
print("chairi: ", chairi_sum)

print("!!!!!baselines!!!!!!!")
meteor_sum = 0
cider_sum = 0
chairs_sum = 0
object_sum = 0
hallucinate_sum = 0

for i in selected_baseline_results:
    meteor_sum += baseline_result[i]["meteor"]
    cider_sum += baseline_result[i]["cider"]
    chairs_sum += baseline_result[i]["chairs"]
    object_sum += baseline_result[i]["objects_num"]
    hallucinate_sum += baseline_result[i]["hallucinate_num"]

meteor_sum = meteor_sum / len(selected_baseline_results)
cider_sum = cider_sum / len(selected_baseline_results)
chairs_sum = chairs_sum / len(selected_baseline_results)
chairi_sum = hallucinate_sum / object_sum

print("meteor: ", meteor_sum)
print("cider: ", cider_sum)
print("chairs: ", chairs_sum)
print("chairi: ", chairi_sum)


annotation_file_path = "/home/czr/contrast_decoding_LVLMs/eval_dataset/val2014/annotations/captions_val2014.json"
        # with the coco api

coco = COCO(annotation_file_path)

new_halc_caption = []
for cap_pair in halc_generated_captions:
    if cap_pair["image_id"] in selected_results:
        new_halc_caption.append(cap_pair)

print("new_halc_caption", len(new_halc_caption))

coco_res = coco.loadRes(
            new_halc_caption,
        )
coco_eval = COCOEvalCap(coco, coco_res)
coco_eval.params["image_id"] = coco_res.getImgIds()
coco_eval.evaluate()

all_overall_scores = defaultdict(list)
# keep track of the overall scores
for metric, score in coco_eval.eval.items():
    all_overall_scores[metric].append(score)

print("all_overall_scores", all_overall_scores)


new_halc_caption = []
for cap_pair in baseline_generated_captions:
    if cap_pair["image_id"] in selected_baseline_results:
        new_halc_caption.append(cap_pair)

print("new_halc_caption", len(new_halc_caption))

coco_res = coco.loadRes(
            new_halc_caption,
        )
coco_eval = COCOEvalCap(coco, coco_res)
coco_eval.params["image_id"] = coco_res.getImgIds()
coco_eval.evaluate()

all_overall_scores = defaultdict(list)
# keep track of the overall scores
for metric, score in coco_eval.eval.items():
    all_overall_scores[metric].append(score)

print("all_overall_scores", all_overall_scores)