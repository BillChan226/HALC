# the script generates captions for the images in the test set and save the captions
import os
import torch
import argparse
import numpy as np
import random
import torchvision
from tqdm import tqdm
from pycocotools.coco import COCO
import json


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
        "--cfg-path",
        default="./eval_configs/minigpt4_llama2_eval_hallucination.yaml",
        help="path to configuration file.",
    )
    parser_group.add_argument(
        "--gpu-id",
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

    chat = Chat(
        model,
        vis_processor,
        device="cuda:{}".format(args.gpu_id),
        stopping_criteria=stopping_criteria,
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
        "--dataset_name",
        type=str,
        default="coco",
        help="Name of the dataset. Default is 'coco'.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/media/zhuokai/SN850X_4TB/Data/coco/val2014",
        help="Test data directory. Default is '/media/zhuokai/SN850X_4TB/Data/coco/val2014'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_captions/",
        help="Output ditectory for saving test results. Default is './generated_captions/'.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size when generating captions. Default is 64.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2000,
        help="Number of evaluation samples from the dataset. Default is 2000.",
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

    # load program level arguments
    args = parser.parse_args()
    model_name = args.model_name
    dataset_name = args.dataset_name
    data_dir = args.data_dir
    # add model and data dir to output dir
    output_dir = args.output_dir
    batch_size = args.batch_size
    num_samples = args.num_samples
    seed = args.seed
    verbosity = args.verbosity

    # print program level arguments
    if verbosity:
        print("\nmodel_name: ", model_name)
        print("dataset_name: ", dataset_name)
        print("data_dir: ", data_dir)
        print("output_dir: ", output_dir)
        print("batch_size: ", batch_size)
        print("num_samples: ", num_samples)
        print("seed: ", seed)

    # set seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load model
    if model_name == "minigpt4":
        model, CONV_VISION, cfg = initialize_mini_gpt_4(parser)

    if verbosity:
        print(f"\n{model_name} model initialized successfully.")

    # with the coco api
    coco_caps = COCO(os.path.join(data_dir, "annotations/captions_val2014.json"))
    # all the image ids
    img_ids = coco_caps.getImgIds()
    # sample image ids
    sampled_img_ids = random.sample(img_ids, num_samples)

    # generate captions
    all_generated_captions = []
    for i, cur_img_id in enumerate(tqdm(sampled_img_ids, desc="Generating Captions")):
        # current image
        cur_img = coco_caps.loadImgs(cur_img_id)[0]
        # current image path in the data dir
        cur_img_path = os.path.join(data_dir, cur_img["file_name"])

        # construct the conversation
        img_list = []
        model.upload_img(cur_img_path, CONV_VISION, img_list)
        model.encode_img(img_list, 38)  # -1 means the last layer
        # question taken from https://arxiv.org/pdf/2305.10355.pdf
        model.ask("Generate a short caption of the image.", CONV_VISION)
        output_text, _, _ = model.answer(CONV_VISION, img_list)

        # append the generated caption to the list
        all_generated_captions.append(
            {
                "image_id": cur_img_id,
                "caption": output_text,
            }
        )

        # clear the chat
        CONV_VISION.messages = []

    # add data name to output dir
    model_type = cfg.model_cfg.model_type.replace("_", "-")
    output_dir = os.path.join(output_dir, f"{model_name}_{model_type}", dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # save all the generated caption as json
    with open(
        os.path.join(
            output_dir,
            f"{model_name}_{model_type}_{dataset_name}_{num_samples}_generated_captions.json",
        ),
        "w",
    ) as f:
        json.dump(all_generated_captions, f)

    if verbosity:
        print(f"\nGenerated captions saved to {output_dir}.")


if __name__ == "__main__":
    main()
