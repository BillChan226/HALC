import os
import sys
import json
import numpy as np
from . import lm_consistency as LM
from . import im_consistency as IM
from .chair import *


def hallucination_file_to_dict(hallucinated_json):
    hallucination_data = json.load(open(hallucinated_json))
    hallucination_by_imid = {h["image_id"]: h for h in hallucination_data["sentences"]}

    return hallucination_by_imid


def get_sentence_scores_from_hallucination_file(hallucination_file):
    hallucination = json.load(open(hallucination_file))
    return hallucination["overall_metrics"]


def get_consistency(tag, annotation_path, robust=False):
    # Load hallucination dict.  If it does not exist, make it!
    hallucinated_json = "./output/hallucination/hallucinated_words_%s.json" % tag
    sentences = "generated_sentences/%s.json" % tag

    if not os.path.exists(hallucinated_json):
        print("Computing hallucination file for tag %s" % tag)
        sentence_template = "generated_sentences/%s.json"
        _, imids, _ = load_generated_captions(sentence_template % tag)
        evaluator = CHAIR(imids, annotation_path)
        evaluator.get_annotations()
        cap_dict = evaluator.compute_chair(sentence_template % tag)
        save_hallucinated_words(sentence_template % tag, cap_dict)

    hallucination_by_imid = hallucination_file_to_dict(hallucinated_json)

    # LM consistency
    word_to_idx = LM.read_vocab(robust)
    blank_lm_predictions = LM.get_blank_prediction_path(tag)

    lm_consistency = LM.get_lm_consistency(
        hallucination_by_imid, blank_lm_predictions, word_to_idx, quiet=True
    )

    # IM consistency
    predicted_label_dict, gt_label_dict = IM.get_label_dicts(robust)
    im_consistency = IM.get_im_consistency(
        hallucination_by_imid, predicted_label_dict, gt_label_dict
    )

    # get chair scores for completeness
    scores = get_sentence_scores_from_hallucination_file(hallucinated_json)

    return scores["CHAIRi"], lm_consistency, im_consistency


def score_correlation(cap_file, quiet=False):
    caps = json.load(open(cap_file))

    ciders = []
    meteors = []
    spices = []
    hallucinations = []

    for cap in caps["sentences"]:
        info = cap["metrics"]
        meteors.append(info["METEOR"])
        ciders.append(info["CIDEr"])
        spices.append(info["SPICE"]["All"]["f"])
        hallucinations.append(1 - info["CHAIRi"])

    meteors = np.array(meteors)
    ciders = np.array(ciders)
    spices = np.array(spices)
    hallucinations = np.array(hallucinations)

    cider_corr = np.corrcoef(ciders, hallucinations)[1][0]
    meteor_corr = np.corrcoef(meteors, hallucinations)[1][0]
    spice_corr = np.corrcoef(spices, hallucinations)[1][0]

    if not quiet:
        print("CIDEr and hallucination: %0.03f" % cider_corr)
        print("METEOR and hallucination: %0.03f" % meteor_corr)
        print("SPICE and hallucination: %0.03f" % spice_corr)

    return cider_corr, meteor_corr, spice_corr


def predictive_metrics(hallucinated_json_1, hallucinated_json_2):
    """
    Can sentence metrics predict hallucination?  In section 3.4 of paper.
    """

    hallucination_data_1 = json.load(open(hallucinated_json_1))
    hallucination_data_2 = json.load(open(hallucinated_json_2))

    def bin_by_spice(data):
        # bin by spice scores
        spices = []
        hallucinations = []

        for cap in data["sentences"]:
            info = cap["metrics"]
            spices.append(info["SPICE"]["All"]["f"])
            hallucinations.append(info["CHAIRs"])

        hist = []
        for i in range(0, 100, 10):
            idxs = [
                idx
                for idx, spice in enumerate(spices)
                if (spice * 100 >= i) and (spice * 100 < (i + 10))
            ]
            if len(idxs) == 0:
                hist.append(0)
            else:
                hist.append(np.mean([hallucinations[idx] for idx in idxs]))
        return hist

    score_histogram_1 = bin_by_spice(hallucination_data_1)
    score_histogram_2 = bin_by_spice(hallucination_data_2)
    return list(np.array(score_histogram_1) - np.array(score_histogram_2))
