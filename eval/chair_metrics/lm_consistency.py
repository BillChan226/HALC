import numpy as np
import sys
import json
import os
import pickle as pkl
from nltk import word_tokenize
from collections import defaultdict
from pattern.en import singularize
import pdb
import sys
import argparse
from .misc import *


def read_vocab(robust):
    # read vocab
    vocab = pkl.load(open("data/vocab.p", "rb"))
    word_to_idx = defaultdict(lambda: unk_idx)  # word -> ix
    for key, value in zip(vocab.keys(), vocab.values()):
        word_to_idx[value] = int(key)

    return word_to_idx


def softmax(array):
    shift = array - np.max(array)
    return np.exp(shift) / np.sum(np.exp(shift))


def get_blank_prediction_path(tag):
    return "./output/language_model_blank_input/%s/%%d.npy" % tag


def get_lm_consistency(
    hallucination_by_imid, blank_lm_predictions, word_to_idx, quiet=False
):
    word_hallucinated_idxs = 0.0
    word_hallucinated_total = 0.0

    for i, imid in enumerate(sorted(hallucination_by_imid.keys())):
        if not quiet:
            sys.stdout.write("\r%d/%d" % (i, len(hallucination_by_imid.keys())))
        probs = np.load(blank_lm_predictions % int(imid))
        item = hallucination_by_imid[imid]
        caption = item["caption"]

        caption_words = word_tokenize(caption.lower())
        mscoco_words = zip(
            item["hallucination_idxs"],
            [caption_words[i] for i in item["hallucination_idxs"]],
        )

        for mscoco_word in mscoco_words:
            idx, word = mscoco_word
            word = word.split(" ")[0]
            word_probs = softmax(probs[idx, :])
            sorted_objects = np.argsort(word_probs)[::-1]
            word_idx = np.where(sorted_objects == word_to_idx[word])[0][0] + 1
            word_hallucinated_idxs += word_idx
            word_hallucinated_total += 1

    return word_hallucinated_total / word_hallucinated_idxs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", type=str, default="coco/annotations")
    parser.add_argument("--tag", type=str, default="td-fc_beam1_test")
    parser.add_argument("--robust", dest="robust", action="store_true")
    parser.set_defaults(robust=False)
    args = parser.parse_args()

    hallucinated_json = "./output/hallucination/hallucinated_words_%s.json" % args.tag
    hallucination_by_imid = hallucination_file_to_dict(hallucinated_json)
    blank_lm_predictions = get_blank_prediction_path(args.tag)

    word_to_idx = read_vocab(args.robust)
    consistency = get_lm_consistency(
        hallucination_by_imid, blank_lm_predictions, word_to_idx
    )
    print("\nConsistency: %0.04f" % consistency)
