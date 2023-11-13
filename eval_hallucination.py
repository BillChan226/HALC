import os
import sys
import torch
import argparse
import numpy as np
import warnings
import random
from chair_metrics import chair


# The script evaluates LLM hallucination on the test set.
# main function
def main():
    # program level args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        help="Choose between 'chair', or 'pope' for evaluation metric.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input file path to the model results.",
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
        default="./hallucination_eval_results/",
        help="Output ditectory for saving test results. Default is './hallucination_eval_results/'.",
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
    metric = args.metric
    input_path = args.input_path
    seed = args.seed
    dataset_name = args.dataset_name
    data_dir = args.data_dir
    output_dir = args.output_dir
    verbosity = args.verbosity

    # print program level arguments
    if verbosity:
        print("\nmetric: ", metric)
        print("input_path: ", input_path)
        print("dataset_name: ", dataset_name)
        print("data_dir: ", data_dir)
        print("output_dir: ", output_dir)
        print("seed: ", seed)

    # sanity check between caption file and command line arguments
    model_name = input_path.split("/")[-1].split("_")[0]
    model_type = input_path.split("/")[-1].split("_")[1]
    dataset_name_identified = input_path.split("/")[-1].split("_")[2]
    if dataset_name_identified != dataset_name:
        raise Exception(
            f"Dataset name in caption file {dataset_name_identified} does not match command line argument {dataset_name}."
        )
    # update output dir
    output_dir = os.path.join(
        output_dir, metric, f"{model_name}_{model_type}", dataset_name
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # different metrics
    if metric == "chair":
        # annotation path should be under data dir
        annotation_dir = f"{data_dir}/annotations"
        # load the generated captions
        _, imids, _ = chair.load_generated_captions(input_path)
        # initialize CHAIR with generated captions and annotations
        evaluator = chair.CHAIR(imids, annotation_dir)
        evaluator.get_annotations()
        # compute chair metrics
        cap_dict = evaluator.compute_chair(input_path)
        # print metric
        metric_string_ce = chair.print_metrics(cap_dict, quiet=False)
        # save hallucinated words
        # chair.save_hallucinated_words(input_path, cap_dict, output_dir)

    elif metric == "pope":
        pass
    else:
        raise ValueError(f"Invalid metric selection {metric}.")


if __name__ == "__main__":
    main()
