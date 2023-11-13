import sys
import json
import pickle as pkl
import pdb
import numpy as np
from nltk import word_tokenize
from pattern.en import singularize
import nltk
import argparse
from .misc import *


def get_label_dicts(robust=False):
    if robust:
        label_dict = "output/image_classifier/classifier_output_robust.p"
    else:
        label_dict = "output/image_classifier/classifier_output.p"
    predicted_label_dict = pkl.load(open(label_dict, "rb"))
    gt_label_dict = pkl.load(open("data/gt_labels.p", "rb"))

    return predicted_label_dict, gt_label_dict


def get_im_consistency(hallucination_by_imid, predicted_label_dict, gt_label_dict):
    total = 0.0
    scores = 0.0

    for i, imid in enumerate(hallucination_by_imid.keys()):
        item = hallucination_by_imid[imid]
        caption = item["caption"]
        caption_words = word_tokenize(caption.lower())
        mscoco_words = [i[1] for i in item["mscoco_hallucinated_words"]]

        predicted_labels = predicted_label_dict[imid]["predicted_classes"]
        raw_output = predicted_label_dict[imid]["raw_output"]
        raw_output_sorted = np.argsort(raw_output)[::-1]

        for mscoco_word in mscoco_words:
            value = raw_output[gt_label_dict["cat_to_idx"][mscoco_word]]
            scores += value
            total += 1

    return scores / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", type=str, default="coco/annotations")
    parser.add_argument("--tag", type=str, default="td-fc_beam1_test")
    parser.add_argument("--robust", dest="robust", action="store_true")
    parser.set_defaults(robust=False)
    args = parser.parse_args()

    # read hallucination file
    hallucinated_json = "./output/hallucination/hallucinated_words_%s.json" % args.tag
    hallucination_by_imid = hallucination_file_to_dict(hallucinated_json)

    predicted_label_dict, gt_label_dict = get_label_dicts(args.robust)
    consistency = get_im_consistency(
        hallucination_by_imid, predicted_label_dict, gt_label_dict
    )
    print("Im consistency is: %0.04f" % consistency)
