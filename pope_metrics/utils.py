import os
import random
import json


def get_image(img_path, seg_num):
    img_list = [os.path.join(img_path, img) for img in os.listdir(img_path)]
    sampled_images = random.sample(img_list, seg_num)
    return sampled_images


def create_question(question_id, image, object_name, label, template):
    # initialize output
    question = dict()
    question["question_id"] = question_id
    question["image"] = image

    # a/an
    template1 = template
    template2 = template.replace("a", "an")
    if object_name[0] not in ["a", "e", "i", "o", "u"]:
        question["text"] = template1.format(object_name)
    elif object_name[0] in ["a", "e", "i", "o", "u"]:
        question["text"] = template2.format(object_name)

    # positive ("yes") or negative ("no")
    question["label"] = label

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
