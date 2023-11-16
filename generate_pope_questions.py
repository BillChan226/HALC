import argparse
import json
import os
import random
from pope_metrics.utils import generate_ground_truth_objects, pope


# main function
def main():
    # program level args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        default=None,
        help="Type of POPE negative sampling, choose between random, popular or adversarial. Default is None. (all three types will be generated)",
    )
    parser.add_argument(
        "--gt_seg_path",
        type=str,
        required=True,
        help="Input json file that contains ground truth objects in the image.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of positive/negative objects to be sampled. Default is 3.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=500,
        help="Number of images to build POPE. Default is 500.",
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
        default="./generated_pope_questions/",
        help="Output ditectory for saving test results. Default is './generated_pope_questions/'.",
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
    pope_type = args.type
    gt_seg_path = args.gt_seg_path
    # get dataset name from gt_seg_path
    dataset_name = gt_seg_path.split("/")[-1].split("_")[0]
    num_samples = args.num_samples
    num_images = args.num_images
    question_template = args.question_template
    output_dir = args.output_dir
    seed = args.seed
    verbosity = args.verbosity

    # print program level arguments
    if verbosity:
        print("\npope_type: ", pope_type)
        print(f"dataset_name: {dataset_name}")
        print(f"seg_path: {gt_seg_path}")
        print(f"num_samples: {num_samples}")
        print(f"num_images: {num_images}")
        print(f"question_template: {question_template}")
        print(f"seed: {seed}")
        print(f"output_dir: {output_dir}")
        print(f"verbosity: {verbosity}")

    # set seed
    random.seed(seed)

    # load ground truth segmentation results.
    # Minimum example (other keys such as image_id can exist):
    # {"image": "COCO_val2014_000000131089.jpg", "objects": ["person", "baseball bat"]}
    segment_results = [json.loads(q) for q in open(gt_seg_path, "r")]
    print(len(segment_results))

    # process segmentation ground truth
    processed_segment_results = []
    # Sample images which contain more than sample_num objects
    for cur_image in segment_results:
        if len(cur_image["objects"]) >= num_samples:
            processed_segment_results.append(cur_image)

    assert (
        len(processed_segment_results) >= num_images
    ), f"The number of images that contain more than {num_samples} objects is less than {num_images}."

    # Randomly sample img_num images
    processed_segment_results = random.sample(processed_segment_results, num_images)

    # Organize the ground truth objects and their co-occurring frequency
    output_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ground truth object summary
    ground_truth_objects = generate_ground_truth_objects(
        processed_segment_results,
        output_dir,
        dataset_name,
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
                output_dir=output_dir,
                dataset_name=dataset_name,
                verbosity=verbosity,
            )
    else:
        pope(
            ground_truth_objects=ground_truth_objects,
            segment_results=processed_segment_results,
            num_samples=num_samples,
            template=question_template,
            neg_strategy=pope_type,
            output_dir=output_dir,
            dataset_name=dataset_name,
            verbosity=verbosity,
        )


if __name__ == "__main__":
    main()
