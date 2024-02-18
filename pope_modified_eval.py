import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import json
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image

from pope_loader import POPEDataSet
from minigpt4.common.dist_utils import get_rank
from minigpt4.models import load_preprocess

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from types import SimpleNamespace
from decoder_zoo.Woodpecker.vis_corrector import Corrector
from decoder_zoo.Woodpecker.config import woodpecker_args_dict
from decoder_zoo.HALC.context_density.halc import halc_assistant
from decoder_zoo.VCD.vcd_utils.vcd_add_noise import add_diffusion_noise

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from collections import defaultdict

from pope_metrics.utils import generate_ground_truth_objects
import os
import random
import json


def get_image(img_path, seg_num):
    img_list = [os.path.join(img_path, img) for img in os.listdir(img_path)]
    sampled_images = random.sample(img_list, seg_num)
    return sampled_images


def create_question(question_id, image, image_id, object_name, label, template):
    # initialize output
    question = dict()
    question["question_id"] = question_id
    question["image"] = image
    question["image_id"] = image_id

    # a/an
    # template1 = template
    # template2 = template.replace("a", "an")
    # if object_name[0] not in ["a", "e", "i", "o", "u"]:
    #     question["text"] = template1.format(object_name)
    # elif object_name[0] in ["a", "e", "i", "o", "u"]:
    #     question["text"] = template2.format(object_name)
    question["text"] = object_name
    # positive ("yes") or negative ("no")
    # question["label"] = True if label == "yes" else False
    question["label"] = label
    # print("question", question)
    # input("here")

    return question


def pope(
    ground_truth_objects,
    segment_results,
    num_samples,
    template,
    neg_strategy,
    output_dir,
    dataset_name,
    verbosity,
):
    # all the questions
    question_list = []
    # question id starts at 1
    question_id = 1
    output_file = os.path.join(
        output_dir, dataset_name + "_pope_" + neg_strategy + "_questions.json"
    )

    # all the ground truth objects
    gt_objects_list = list(ground_truth_objects.keys())
    # sort the ground truth objects by their frequency
    sorted_objects = sorted(
        ground_truth_objects.items(), key=lambda x: x[1], reverse=True
    )
    # compute co-occurrence from the ground truth segmentation results
    # {object1: [object2, object3, ...], ...}
    sorted_co_occur = compute_co_occurrence(
        segment_results,
        output_dir,
        dataset_name,
        verbosity,
    )

    # for each image
    for cur_image in segment_results:
        # all the sampled objects
        history_object_list = []

        # Positive sampling
        for i in range(num_samples):
            cur_pos_object_name = cur_image["objects"][i]
            history_object_list.append(cur_pos_object_name)
            # create the question (dict)
            question = create_question(
                question_id=question_id,
                image=cur_image["image"],
                image_id=cur_image["image_id"],
                object_name=cur_pos_object_name,
                label="yes",  # positive
                template=template,
            )
            question_list.append(question)
            question_id += 1

            # Negative sampling (random)
            if neg_strategy == "random":
                # randomly select an object
                cur_neg_object_name = random.choice(gt_objects_list)
                # make sure the selected object is not in the history list or the current image
                while (
                    cur_neg_object_name in history_object_list
                    or cur_neg_object_name in cur_image["objects"]
                ):
                    cur_neg_object_name = random.choice(gt_objects_list)
                history_object_list.append(cur_neg_object_name)
                question = create_question(
                    question_id=question_id,
                    image=cur_image["image"],
                    image_id=cur_image["image_id"],
                    object_name=cur_neg_object_name,
                    label="no",  # negative
                    template=template,
                )
                question_list.append(question)
                question_id += 1

            # Negative sampling (popular)
            elif neg_strategy == "popular":
                flag = 0
                # for each object in the sorted object list
                for j in range(len(sorted_objects)):
                    cur_neg_object_name = sorted_objects[j][0]
                    if (
                        cur_neg_object_name not in history_object_list
                        and cur_neg_object_name not in cur_image["objects"]
                    ):
                        history_object_list.append(cur_neg_object_name)
                        question = create_question(
                            question_id=question_id,
                            image=cur_image["image"],
                            image_id=cur_image["image_id"],
                            object_name=cur_neg_object_name,
                            label="no",  # negative
                            template=template,
                        )
                        question_list.append(question)
                        question_id += 1
                        flag = 1
                        break

                # In case no object is selected, randomly select an object
                if not flag:
                    while True:
                        cur_neg_object_name = random.choice(gt_objects_list)
                        if (
                            cur_neg_object_name not in history_object_list
                            and cur_neg_object_name not in cur_image["objects"]
                        ):
                            history_object_list.append(cur_neg_object_name)
                            question = create_question(
                                question_id=question_id,
                                image=cur_image["image"],
                                image_id=cur_image["image_id"],
                                object_name=cur_neg_object_name,
                                label="no",  # negative
                                template=template,
                            )
                            question_list.append(question)
                            question_id += 1
                            break

            # Negative sampling (Adversarial)
            elif neg_strategy == "adversarial":
                flag = 0
                for j in range(len(sorted_co_occur[cur_pos_object_name])):
                    # select the object that co-occurs the most with the current object
                    cur_neg_object_name = sorted_co_occur[cur_pos_object_name][j]
                    if (
                        cur_neg_object_name not in history_object_list
                        and cur_neg_object_name not in cur_image["objects"]
                    ):
                        history_object_list.append(cur_neg_object_name)
                        question = create_question(
                            question_id=question_id,
                            image=cur_image["image"],
                            image_id=cur_image["image_id"],
                            object_name=cur_neg_object_name,
                            label="no",  # negative
                            template=template,
                        )
                        question_list.append(question)
                        question_id += 1
                        flag = 1
                        break

                # In case no object is selected, randomly select an object
                if not flag:
                    while True:
                        cur_neg_object_name = random.choice(gt_objects_list)
                        if (
                            cur_neg_object_name not in history_object_list
                            and cur_neg_object_name not in cur_image["objects"]
                        ):
                            history_object_list.append(cur_neg_object_name)
                            question = create_question(
                                question_id=question_id,
                                image=cur_image["image"],
                                image_id=cur_image["image_id"],
                                object_name=cur_neg_object_name,
                                label="no",  # negative
                                template=template,
                            )
                            question_list.append(question)
                            question_id += 1
                            break
            else:
                raise Exception(f"Invalid negative sampling strategy {neg_strategy}.")

    with open(output_file, "w") as f:
        for question in question_list:
            json_str = json.dumps(question)
            f.write(json_str + "\n")

    if verbosity:
        print("\nPOPE pos/neg questions saved to ", output_file)


# summary of ground truth objects and their frequency
def generate_ground_truth_objects(segment_results, output_dir, dataset_name, verbosity):
    gt_objects = dict()
    output_file = os.path.join(output_dir, dataset_name + "_ground_truth_objects.json")

    for image in segment_results:
        seg = image["objects"]
        for o in seg:
            if o not in gt_objects:
                gt_objects[o] = 1
            else:
                gt_objects[o] += 1

    with open(output_file, "w") as f:
        json_str = json.dumps(gt_objects)
        f.write(json_str)

    if verbosity:
        print("\nGround truth objects saved to ", output_file)

    return gt_objects


def compute_co_occurrence(segment_results, output_dir, dataset_name, verbosity):
    output_file = os.path.join(output_dir, dataset_name + "_co_occur.json")
    co_occur = dict()

    for image in segment_results:
        objects = image["objects"]
        for o in objects:
            if o not in co_occur:
                co_occur[o] = dict()
            for other_o in objects:
                if o == other_o:
                    continue
                if other_o not in co_occur[o]:
                    co_occur[o][other_o] = 1
                else:
                    co_occur[o][other_o] += 1

    sorted_co_occur = dict()
    for o in co_occur:
        objects = co_occur[o]
        sorted_co_occur_objects = sorted(
            objects.items(), key=lambda x: x[1], reverse=True
        )
        sorted_co_occur[o] = [item[0] for item in sorted_co_occur_objects]

    with open(output_file, "w") as f:
        json_str = json.dumps(sorted_co_occur)
        f.write(json_str)

    if verbosity:
        print("\nCo-occurrence saved to ", output_file)

    return sorted_co_occur



MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "instructblip": "eval_configs/instructblip_eval.yaml",
    "lrv_instruct": "eval_configs/lrv_instruct_eval.yaml",
    "shikra": "eval_configs/shikra_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
    "mplug-owl2": "eval_configs/mplug-owl2_eval.yaml",
}

POPE_PATH = {
    "random": "pope_coco/coco_pope_random.json",
    "popular": "pope_coco/coco_pope_popular.json",
    "adversarial": "pope_coco/coco_pope_adversarial.json",
}


INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:",
    "mplug-owl2": "USER: <|image|><question> ASSISTANT:",
}


def parse_args():
    parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
    parser.add_argument("--model", type=str, default="mplug-owl2", help="model")
    parser.add_argument("--pope_type", type=str, help="model")
    # parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="specify the gpu to load the model."
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="coco",
        help="Name of the dataset. Default is 'coco'.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/czr/contrast_decoding_LVLMs/eval_dataset/val2014/",
        help="data path",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="num workers")

    parser.add_argument("-b", "--beam", type=int, default=1)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--scale_factor", type=float, default=50)
    parser.add_argument("--threshold", type=int, default=15)
    parser.add_argument("--num_attn_candidates", type=int, default=5)
    parser.add_argument("--penalty_weights", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-m", "--max_new_tokens", type=int, default=16)
    parser.add_argument(
        "-d",
        "--decoder",
        type=str,
        default="greedy",
        help="Decoding strategy to use. You can choose from 'greedy', 'dola', 'halc'. Default is 'greedy'.",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        action="store_false",
        dest="verbosity",
        default=True,
        help="Verbosity. Default: True.",
    )
    parser.add_argument(
        "-k",
        "--k-candidate-num",
        type=int,
        default=4,
        help="specify the k candidate number for halc.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./paper_result/",
        help="Output ditectory for saving test results. Default is './generated_chair_inputs/'.",
    )
    parser.add_argument(
        "-p",
        "--post-correction",
        type=str,
        default=None,
        help="Post correction method such as woodpecker, lure.",
    )
    parser.add_argument(
        "-e",
        "--expand-ratio",
        type=float,
        default=0.6,
        help="Expand ratio of growing contextual field.",
    )
    parser.add_argument(
        "--cd_alpha",
        type=float,
        default=1,
        help="Alpha param for VCD.",
    )
    parser.add_argument(
        "--cd_beta", type=float, default=0.1, help="Beta param for VCD."
    )
    parser.add_argument(
        "--noise_step", type=int, default=500, help="Noise step for VCD."
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="dino",
        help="Detector type. Default is 'groundingdino'.",
    )
    parser.add_argument(
        "--gt_seg_path",
        type=str,
        default="pope_coco/coco_ground_truth_segmentation.json",
        help="Input json file that contains ground truth objects in the image.",
    )
    parser.add_argument(
        "-n",
        "--num_images",
        type=int,
        default=100,
        help="Number of images to build POPE questions. Default is 500.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of positive/negative objects to be sampled. Default is 3.",
    )
    parser.add_argument(
        "--question_template",
        type=str,
        default="Is there a {} in the image? ",
        # default="Is there a XXX in the image? There is no XXX in the image, so the answer is No. Is there a YYY in the image? There is 2 YYY in the image, so the answer is Yes. Is there a {} in the image? ",
        # default="Find evidence first and then answer: is there a {} in the image?",
        # default="Is there a {} in the image?",  # for llava-1.5
        help="Prompt template. Default is 'Is there a {} in the image?'.",
    )
    parser.add_argument(
        "-c",
        "--generated_caption",
        type=str,
        default=None,
        help="Directory of the generated captions.",
    )

    args = parser.parse_args()
    return args


def setup_seeds(config, seed):
    # seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def print_acc(pred_list, label_list):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)
    # unknown_ratio = pred_list.count(2) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print("TP\tFP\tTN\tFN\t")
    print("{}\t{}\t{}\t{}".format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print("Accuracy: {}".format(acc))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1 score: {}".format(f1))
    print("Yes ratio: {}".format(yes_ratio))

    return acc, precision, recall, f1


def recorder(out, pred_list, text):
    # NEG_WORDS = ["No", "not", "no", "NO"]
    # for line in out:
    #     line = line.replace(".", "")
    #     line = line.replace(",", "")
    #     words = line.split(" ")
        # if any(word in NEG_WORDS for word in words) or any(
        #     word.endswith("n't") for word in words
        # ):
        #     pred_list.append(0)
        # else:
        #     pred_list.append(1)
    # print("text", text)
    text = text[0].split(" ")
    # label = str(label.cpu().numpy().tolist())
    # if obj in out for obj in text:
    #     pred_list.append(1)
    # else:
    #     pred_list.append(0)
    flag = False
    for obj in text:
        if obj in out:
            pred_list.append(1)
            flag = True
            break

    if not flag:
        pred_list.append(0)

    # print("out", out)
    # print("text", text)
    # print("pred_list", pred_list)
    # input()


    return pred_list


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
    args.pope_path = POPE_PATH[args.pope_type]
    cfg = Config(args)

    decoding_strategy = args.decoder
    seed = args.seed
    setup_seeds(cfg, seed)
    pope_type = args.pope_type
    device = (
        torch.device(f"cuda:{int(args.gpu_id)}") if torch.cuda.is_available() else "cpu"
    )
    model_name = args.model
    verbosity = args.verbosity
    k_candidate_num = args.k_candidate_num
    detector_type = args.detector
    num_samples = args.num_samples
    num_images = args.num_images
    dataset_name = args.dataset_name
    data_path = args.data_path
    output_dir = args.output_dir
    num_beams = args.beam
    num_workers = args.num_workers
    batch_size = args.batch_size
    post_correction = args.post_correction
    max_new_tokens = args.max_new_tokens
    expand_ratio = args.expand_ratio
    cd_alpha = args.cd_alpha
    cd_beta = args.cd_beta
    gt_seg_path = args.gt_seg_path
    question_template = args.question_template
    generated_caption_path = args.generated_caption
    loaded_json = []
    loaded_keys = []
    loaded_dict = {}
    with open(generated_caption_path, "r") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            # print("idx", idx)
            pair = json.loads(line)
            loaded_json.append(pair)
            loaded_dict[pair["image_id"]] = pair["caption"]
            loaded_keys.append(pair["image_id"])
    # device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # ========================================
    #             Model Initialization
    # ========================================
    print("Initializing Model")

    # model_config = cfg.model_cfg
    # model_config.device_8bit = args.gpu_id
    # model_cls = registry.get_model_class(model_config.arch)
    # model = model_cls.from_config(model_config).to(device)
    # model.eval()
    vis_processors, txt_processors = load_preprocess(cfg.get_config().preprocess)
    # vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    # vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
    #     vis_processor_cfg
    # )
    # # vis_processors.do_normalize = False
    # print(vis_processors["eval"].transform)

    valid_decoding_strategies = [
        "greedy",
        "dola",
        "halc-dola",
        "halc-greedy",
        "halc-beam",
        "opera-beam",
        "vcd",
    ]
    valid_post_editing_strategies = ["lure", "woodpecker"]

    assert (
        decoding_strategy in valid_decoding_strategies
    ), f"Invalid decoding strategy: {decoding_strategy}, should be in {valid_decoding_strategies}"
    assert (
        post_correction in valid_post_editing_strategies or post_correction is None
    ), f"Invalid post correction strategy: {post_correction}, should be in {valid_post_editing_strategies}"

    decoding_strategy = decoding_strategy
    opera_decoding = False
    dola_decoding = False
    halc_decoding = False
    vcd_decoding = False
    beam_search = False

    print("decoding_strategy", decoding_strategy)
    if decoding_strategy == "greedy":
        pass
    elif decoding_strategy == "dola":
        dola_decoding = True

    elif decoding_strategy == "halc-dola":
        dola_decoding = True
        halc_decoding = True
    elif decoding_strategy == "halc-greedy":
        halc_decoding = True
    elif decoding_strategy == "halc-beam":
        halc_decoding = True
        dola_decoding = True
        beam_search = True
    elif decoding_strategy == "opera-beam":
        beam_search = True
        opera_decoding = True
    elif decoding_strategy == "vcd":
        vcd_decoding = True

    if verbosity:
        print("\ndecoding strategy: ", decoding_strategy)
        print("backbone model_name: ", args.model)
        print("data_path: ", data_path)
        print("output_dir: ", output_dir)
        print("num_samples: ", num_samples)
        print("num_images: ", num_images)
        print("num_beams: ", num_beams)
        print("seed: ", seed)
        # print(vis_processors["eval"].transform)

    print("Done!")

    if verbosity:
        print(f"\nGenerating {pope_type} POPE questions")

    # generate pope questions
    question_dir = os.path.join(output_dir, "pope")
    if not os.path.exists(question_dir):
        os.makedirs(question_dir)
    question_path = os.path.join(
        question_dir,
        f"_num_images_{num_images}_num_samples_{num_samples}_pope_{pope_type}_questions.json",
    )
    # load ground truth segmentation results.
    # Must include (other keys such as image_id can exist):
    # {"image": "COCO_val2014_000000131089.jpg", "objects": ["person", "baseball bat"]}
    segment_results = [json.loads(q) for q in open(gt_seg_path, "r")]
    if verbosity:
        print(
            f"\nGround truth segmentation results loaded successfully, contains {len(segment_results)} classes."
        )

    # process segmentation ground truth
    processed_segment_results = []
    # # Sample images which contain more than sample_num objects
    for cur_image in segment_results:
        if len(cur_image["objects"]) >= num_samples:
            processed_segment_results.append(cur_image)

    assert (
        len(processed_segment_results) >= num_images
    ), f"The number of images that contain more than {num_samples} objects is less than {num_images}."

    # Randomly sample num_images images
    # print("processed_segment_results", len(processed_segment_results))
    # extract those with the same image_id as loaded_keys from processed_segment_results
    processed_segment_results = [
        item for item in processed_segment_results if item["image_id"] in loaded_keys
    ]
    # processed_segment_results = random.sample(processed_segment_results, num_images)
    print("processed_segment_results", len(processed_segment_results))

    # Organize the ground truth objects and their co-occurring frequency
    question_name = f"_num_images_{num_images}_num_samples_{num_samples}"
    # ground truth object summary
    ground_truth_objects = generate_ground_truth_objects(
        processed_segment_results,
        question_dir,
        question_name,
        verbosity,
    )

    # print("ground_truth_objects", ground_truth_objects)
    # input()

    # Generate POPE questions and save to local file
    if pope_type is None:
        for cur_type in ["random", "popular", "adversarial"]:
            pope(
                ground_truth_objects=ground_truth_objects,
                segment_results=processed_segment_results,
                num_samples=num_samples,
                template=question_template,
                neg_strategy=cur_type,
                output_dir=question_dir,
                dataset_name=question_name,
                verbosity=verbosity,
            )
    else:
        pope(
            ground_truth_objects=ground_truth_objects,
            segment_results=processed_segment_results,
            num_samples=num_samples,
            template=question_template,
            neg_strategy=pope_type,
            output_dir=question_dir,
            dataset_name=question_name,
            verbosity=verbosity,
        )

    # load all the POPE questions
    all_pope_questions = [json.loads(q) for q in open(question_path, "r")]
    if verbosity:
        print(
            f"\nLoaded {len(all_pope_questions)} POPE questions from {question_path}."
        )
    # # sanity check
    # if len(all_pope_questions) != num_images * num_samples * 2:
    #     raise ValueError(
    #         f"Number of POPE questions loaded from {question_path} is not equal to {num_images * num_samples * 2}."
    #     )

    # print("all_pope_questions", all_pope_questions)
    # save all the POPE questions to local file
    # if not os.path.exists(question_dir):
    #     os.makedirs(pope_question_dir)
    # pope_question_path = os.path.join(
    #     pope_question_dir,
    #     f"_num_images_{num_images}_num_samples_{num_samples}_pope_{pope_type}_questions.json",
    # )
    # input()

    # load pope data
    pope_dataset = POPEDataSet(
        pope_path=question_path, data_path=args.data_path, trans=vis_processors["eval"]
    )
    pope_loader = torch.utils.data.DataLoader(
        pope_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    print("load data finished")

    base_dir = os.path.join(output_dir, "pope", args.model)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # halc_params = {
    #     "context_domain": "upper",
    #     "contrast_weight": 0.05,
    #     "context_window": 4,
    #     "expand_ratio": expand_ratio,
    #     "beam_size": num_beams,
    #     "k_candidate_num": args.k_candidate_num,
    #     "LVLM_backbone": model_name,
    #     "detector": detector_type,
    # }
    # halc_assistant_helper = halc_assistant(
    #     model,
    #     vis_processor=vis_processor,
    #     device=device,
    #     halc_params=halc_params,
    #     max_new_tokens=max_new_tokens,
    # )

    # lm_early_exit_layers = [
    #     0,
    #     2,
    #     4,
    #     6,
    #     8,
    #     10,
    #     12,
    #     14,
    #     16,
    #     18,
    #     20,
    #     22,
    #     24,
    #     26,
    #     28,
    #     30,
    #     32,
    # ]

    # mature_layer = lm_early_exit_layers[-1]
    # premature_layer = None
    # candidate_premature_layers = lm_early_exit_layers[:-1]
    # premature_layer_dist = {l: 0 for l in candidate_premature_layers}

    print("Start eval...")
    pred_list, pred_list_s, label_list = [], [], []

    for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
        image = data["image"]
        qu = data["query"]
        label = data["label"]
        object_text = data["query"]
        
        image_path = data["image_path"]
        image_id = image_path[0].split("/")[-1].split(".")[0].split("_")[-1].lstrip("0")
        label_list = label_list + list(label)

        template = INSTRUCTION_TEMPLATE[args.model]
        qu = [template.replace("<question>", q) for q in qu][0]

        image = image.to(device)
        label = torch.Tensor(label).to(device)

        out = loaded_dict[int(image_id)]
        # print("out", out)
        # input()
        pred_list = recorder(out, pred_list, object_text)
        # for line in out:
        #     print(line)

        # output_text = out#[0]
        # cur_generated_answer = {
        #     "image_id": image_id,
        #     "question": " ".join(qu[0].split(" ")[2:]).split("?")[0] + "?",
        #     "answer": output_text,
        # }

        # # dump metric file
        # generated_captions_path = os.path.join(
        #     base_dir,
        #     f"{model_name}_{decoding_strategy}_beams_{num_beams}_k_{k_candidate_num}_{dataset_name}_expand_ratio_{expand_ratio}_seed_{seed}_max_tokens_{max_new_tokens}_samples_{num_images}_pope_{pope_type}_generated_captions.json",
        # )
        # with open(generated_captions_path, "a") as f:
        #     json.dump(cur_generated_answer, f)
        #     f.write("\n")

    # print(
    #     "[{}, {}]===============================================".format(
    #         args.scale_factor, args.num_attn_candidates
    #     )
    # )
    if len(pred_list) != 0:
        acc, precision, recall, f1 = print_acc(pred_list, label_list)
    if len(pred_list_s) != 0:
        acc, precision, recall, f1 = print_acc(pred_list_s, label_list)

    result = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    }

    # print to a CSV
    # csv_path = "/home/czr/HaLC/eval/eval_pope_results.csv"
    # if not os.path.exists(csv_path):
    #     with open(csv_path, "w") as f:
    #         f.write("model,seed,accuracy,precision,recall,f1\n")
    # with open(csv_path, "a") as f:
    #     f.write(
    #         f"{model_name},{seed},{acc},{precision},{recall},{f1}\n"
    #     )
    
        

    # metrics_path = os.path.join(
    #     base_dir,
    #     f"{model_name}_{decoding_strategy}_beams_{num_beams}_k_{k_candidate_num}_{dataset_name}_expand_ratio_{expand_ratio}_seed_{seed}_max_tokens_{max_new_tokens}_samples_{num_images}_pope_{pope_type}_results.json",
    # )
    # with open(metrics_path, "w") as f:
    #     json.dump(result, f)
    #     f.write("\n")


if __name__ == "__main__":
    main()
