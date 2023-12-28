import os
import argparse
import json
from chair_metrics import chair


# The script evaluates LLM hallucination on the test set.
# main function
def main():
    # program level args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        type=str,
        required=False,
        default="chair",
        help="Choose between 'chair', or 'pope' for evaluation metric.",
    )
    parser.add_argument(
        "--chair_input_path",
        type=str,
        help="Input file path to the model CHAIR results.",
    )
    parser.add_argument(
        "--pope_answer_path",
        type=str,
        help="Input file path to the model POPE answers.",
    )
    parser.add_argument(
        "--pope_question_path",
        type=str,
        help="Input file path to the data POPE questions.",
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
        default="/home/czr/contrast_decoding_LVLMs/eval_dataset/val2014",

        help="Test data directory. Default is 'eval_dataset/val2014'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./hallucination_results",
        help="Test data directory. Default is './hallucination_results'.",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        action="store_false",
        dest="verbosity",
        default=True,
        help="Verbosity. Default: True.",
    )

    # load program level arguments
    args = parser.parse_args()
    metric = args.metric
    dataset_name = args.dataset_name
    data_dir = args.data_dir
    output_dir = args.output_dir
    verbosity = args.verbosity

    # print program level arguments
    if verbosity:
        print("\nmetric: ", metric)
        print("dataset_name: ", dataset_name)
        print("data_dir: ", data_dir)
        print("output_dir: ", output_dir)

    # different metrics
    if metric == "chair":
        chair_input_path = args.chair_input_path
        if verbosity:
            print("\nchair_input_path: ", chair_input_path)

        # sanity check between caption file and command line arguments
        model_name = chair_input_path.split("/")[-1].split("_")[0]
        model_type = (
            chair_input_path.split("/")[-1].split("_")[1]
            + "_"
            + chair_input_path.split("/")[-1].split("_")[2]
        )
        num_images = chair_input_path.split("/")[-1].split("_")[-2]
        # dataset_name_identified = chair_input_path.split("/")[-1].split("_")[-4]

        # if dataset_name_identified != dataset_name:
        #     raise Exception(
        #         f"Dataset name in caption file {dataset_name_identified} does not match command line argument {dataset_name}."
        #     )
        # update output dir
        output_dir = os.path.join(
            output_dir, metric, f"{model_name}_{model_type}", dataset_name
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # annotation path should be under data dir
        annotation_dir = f"{data_dir}/annotations"
        # load the generated captions
        _, imids, _ = chair.load_generated_captions(chair_input_path)
        # initialize CHAIR with generated captions and annotations
        evaluator = chair.CHAIR(imids, annotation_dir)
        evaluator.get_annotations()
        # compute chair metrics
        cap_dict = evaluator.compute_chair(chair_input_path)
        # save to json pretty print
        chair_json_path = os.path.join(
            output_dir,
            f"{model_name}_{model_type}_{dataset_name}_num_images_{num_images}_chair_results.json",
        )
        with open(chair_json_path, "w") as f:
            json.dump(cap_dict, f, indent=4)
        # print metric
        metric_string_ce = chair.print_metrics(cap_dict, quiet=False)
        
        # save results
        result_path = os.path.join(
            output_dir,
            f"{model_name}_{model_type}_{dataset_name}_num_images_{num_images}_chair_results.txt",
        )
        with open(result_path, "w") as f:
            f.write(metric_string_ce)
        if verbosity:
            print(f"\nCHAIR results saved to {result_path}.")
        # save hallucinated words
        # chair.save_hallucinated_words(chair_input_path, cap_dict, output_dir)

    elif metric == "pope":
        pope_answer_path = args.pope_answer_path
        pope_question_path = args.pope_question_path
        if verbosity:
            print("\npope_answer_path: ", pope_answer_path)
            print("pope_question_path: ", pope_question_path)

        answers = [json.loads(q) for q in open(pope_answer_path, "r")][0]
        label_list = [json.loads(q)["label"] for q in open(pope_question_path, "r")]

        for answer in answers:
            text = answer["answer"]

            # Only keep the first sentence
            if text.find(".") != -1:
                text = text.split(".")[0]

            text = text.replace(",", "")
            words = text.split(" ")
            if "No" in words or "not" in words or "no" in words:
                answer["answer"] = "no"
            else:
                answer["answer"] = "yes"

        for i in range(len(label_list)):
            if label_list[i] == "no":
                label_list[i] = 0
            else:
                label_list[i] = 1

        pred_list = []
        for answer in answers:
            if answer["answer"] == "no":
                pred_list.append(0)
            else:
                pred_list.append(1)

        pos = 1
        neg = 0
        yes_ratio = pred_list.count(1) / len(pred_list)

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
    else:
        raise ValueError(f"Invalid metric selection {metric}.")


if __name__ == "__main__":
    main()
