# the script generates captions for the images in the test set and save the captions
import os
import torch
import argparse
import numpy as np
import random
from tqdm import tqdm
import json
from pope_metrics.utils import generate_ground_truth_objects, pope


def initialize_mini_gpt_4(parser):
    from transformers import StoppingCriteriaList
    from minigpt4.conversation.conversation import (
        Chat,
        CONV_VISION_Vicuna0,
        CONV_VISION_LLama2,
        StoppingCriteriaSub,
    )
    from minigpt4.common.config import Config
    from minigpt4.common.registry import registry

    # model specific parser
    parser_group = parser.add_argument_group("MiniGPT4")
    parser_group.add_argument(
        "--cfg_path",
        default="./eval_configs/minigpt4_llama2_eval_hallucination.yaml",
        help="path to configuration file.",
    )
    parser_group.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="specify the gpu to load the model.",
    )
    parser_group.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()

    decoding_strategy = args.decoder
    # load config
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id  # 0
    model_cls = registry.get_model_class(model_config.arch)  # minigpt4
    model = model_cls.from_config(model_config).to("cuda:{}".format(args.gpu_id))

    # available models
    conv_dict = {
        "pretrain_vicuna0": CONV_VISION_Vicuna0,
        "pretrain_llama2": CONV_VISION_LLama2,
    }
    CONV_VISION = conv_dict[model_config.model_type]

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
        vis_processor_cfg
    )

    stop_words_ids = [[835], [2277, 29937]]
    stop_words_ids = [
        torch.tensor(ids).to(device="cuda:{}".format(args.gpu_id))
        for ids in stop_words_ids
    ]
    stopping_criteria = StoppingCriteriaList(
        [StoppingCriteriaSub(stops=stop_words_ids)]
    )

    halc_params = {"context_domain": "upper", "contrast_weight": 0.05, "context_window": 4, "expand_ratio": 0.15}
    hyper_params = {"halc_params": halc_params}

    chat = Chat(
        model,
        vis_processor,
        device="cuda:{}".format(args.gpu_id),
        stopping_criteria=stopping_criteria,
        decoding_strategy = decoding_strategy,
        hyper_params=hyper_params,
    )

    return chat, CONV_VISION, cfg


# main function
def main():
    # program level args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="minigpt4",
        help="Name of the model. Default is 'minigpt4'.",
    )
    parser.add_argument(
        "-d",
        "--decoder",
        type=str,
        default="greedy",
        help="Decoding strategy to use. You can choose from 'greedy', 'dola', 'halc'. Default is 'greedy'.",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="random",
        help="Type of POPE negative sampling, choose between random, popular or adversarial. Default is random.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="coco",
        help="Name of the dataset. Default is 'coco'.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="eval_dataset/val2014",
        help="Test data directory. Default is '/media/zhuokai/SN850X_4TB/Data/coco/val2014'.",
    )
    parser.add_argument(
        "--gt_seg_path",
        type=str,
        default="./pope_metrics/segmentation/coco_ground_truth_segmentation.json",
        help="Input json file that contains ground truth objects in the image.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=500,
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
        default="Is there a {} in the image?",
        help="Prompt template. Default is 'Is there a {} in the image?'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_pope_inputs/",
        help="Output ditectory for saving test results. Default is './generated_pope_inputs/'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Set universal seed. Default is 1.",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        action="store_true",
        dest="verbosity",
        default=False,
        help="Verbosity. Default: False.",
    )
    parser.add_argument(
        "-g",
        "--gpu-id",
        type=int,
        default=0,
        help="specify the gpu to load the model.",
    )

    # load program level arguments
    args = parser.parse_args()
    model_name = args.model_name
    pope_type = args.type
    dataset_name = args.dataset_name
    data_dir = args.data_dir
    gt_seg_path = args.gt_seg_path
    num_images = args.num_images
    num_samples = args.num_samples
    question_template = args.question_template
    output_dir = args.output_dir
    seed = args.seed
    verbosity = args.verbosity
    decoding_strategy = args.decoder

    # print program level arguments
    if verbosity:
        print("\nmodel_name: ", model_name)
        print("pope_type: ", pope_type)
        print(f"dataset_name: {dataset_name}")
        print(f"data_dir: {data_dir}")
        print(f"seg_path: {gt_seg_path}")
        print(f"num_images: {num_images}")
        print(f"num_samples: {num_samples}")
        print(f"question_template: {question_template}")
        print(f"output_dir: {output_dir}")
        print(f"seed: {seed}")

    # set seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # generate pope questions
    question_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(question_dir):
        os.makedirs(question_dir)
    question_path = os.path.join(
        question_dir,
        f"{dataset_name}_num_images_{num_images}_num_samples_{num_samples}_pope_{pope_type}_questions.json",
    )

    # generate pope questions if not exists
    if not os.path.exists(question_path):
        if verbosity:
            print(f"\nGenerating {pope_type} POPE questions")

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
        # Sample images which contain more than sample_num objects
        for cur_image in segment_results:
            if len(cur_image["objects"]) >= num_samples:
                processed_segment_results.append(cur_image)

        assert (
            len(processed_segment_results) >= num_images
        ), f"The number of images that contain more than {num_samples} objects is less than {num_images}."

        # Randomly sample num_images images
        processed_segment_results = random.sample(processed_segment_results, num_images)

        # Organize the ground truth objects and their co-occurring frequency
        question_name = (
            f"{dataset_name}_num_images_{num_images}_num_samples_{num_samples}"
        )
        # ground truth object summary
        ground_truth_objects = generate_ground_truth_objects(
            processed_segment_results,
            question_dir,
            question_name,
            verbosity,
        )

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
    # sanity check
    if len(all_pope_questions) != num_images * num_samples * 2:
        raise ValueError(
            f"Number of POPE questions loaded from {question_path} is not equal to {num_images * num_samples * 2}."
        )

    # load model
    if model_name == "minigpt4":
        model, CONV_VISION, cfg = initialize_mini_gpt_4(parser)

    if verbosity:
        print(f"\n{model_name} model initialized successfully.")

    # # set output dir
    # model_type = cfg.model_cfg.model_type.replace("_", "-")
    # if cfg.model_cfg.dola_decoding is True:
    #     dola_name = "dola"
    #     for layer_index in cfg.model_cfg.early_exit_layers:
    #         dola_name += f"_{layer_index}"
    #     model_type += f"_{dola_name}"
    # output_dir = os.path.join(output_dir, f"{model_name}_{model_type}", dataset_name)

    # set output dir
    model_type = cfg.model_cfg.model_type.replace("_", "-")
    output_dir = os.path.join(
        output_dir, f"{model_name}_{model_type}", decoding_strategy, dataset_name
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # query the model for answers
    all_generated_answers = []
    for i, cur_question in enumerate(
        tqdm(all_pope_questions, desc="Generating answers")
    ):
        # current image path
        cur_img_path = os.path.join(data_dir, cur_question["image"])
        # current query prompt
        cur_prompt = cur_question["text"]

        # construct the conversation
        img_list = []
        model.upload_img(cur_img_path, CONV_VISION, img_list)
        model.encode_img(img_list, 38)  # -1 means the last layer
        # question taken from https://arxiv.org/pdf/2305.10355.pdf
        model.ask(cur_prompt, CONV_VISION)
        output_text, _, _ = model.answer(
            CONV_VISION,
            img_list,
            # dola_decoding=cfg.model_cfg.dola_decoding,
        )

        # append the generated caption to the list
        all_generated_answers.append(
            {
                "question": cur_prompt,
                "answer": output_text,
            }
        )

        # clear the chat
        CONV_VISION.messages = []

    # generated answers path
    generated_answers_path = os.path.join(
        output_dir,
        f"{model_name}_{model_type}_{dataset_name}_num_images_{num_images}_num_samples_{num_samples}_generated_pope_{pope_type}_answers.json",
    )
    with open(generated_answers_path, "w") as f:
        json.dump(all_generated_answers, f)
    if verbosity:
        print(f"\nPOPE answers saved to {generated_answers_path}.")


if __name__ == "__main__":
    main()
