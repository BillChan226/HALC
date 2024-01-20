import os
import re
import subprocess
import csv
import argparse

# Set the directory where the chair.json files are located
# directory = './paper_result/32_tokens/minigpt4/'

parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")

parser.add_argument(
    "-c",
    "--chair-path",
    type=str,
    required=True,
    help="Path to the generated CHAIR captions",
)

args = parser.parse_known_args()[0]

directory = args.chair_path


# Function to run the eval_hallucination command and parse the output
def run_eval(file_path):
    # Running the eval_hallucination command for the given file
    result = subprocess.run(
        ["python", "eval_hallucination.py", "--chair_input_path", file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # print(result.stdout)
    # input()
    # Using regex to extract the metrics from the command output
    # metrics = re.findall(r'SPICE\s*(\d+\.\d+)\s*METEOR\s*(\d+\.\d+)\s*CIDEr\s*(\d+\.\d+)\s*CHAIRs\s*(\d+\.\d+)\s*CHAIRi\s*(\d+\.\d+)',
    #                      result.stdout)
    metrics = re.findall(
        r"\d+\.\d+",
        result.stdout.split("ground truth captions")[-1].split("CHAIR results")[0],
    )

    metrics = metrics if metrics else None

    # Regex patterns to find hallucinate_sum and object_sum
    hallucination_sum_pattern = r"hallucinate_sum:\s+(\d+)"
    object_sum_pattern = r"object_sum:\s+(\d+)"
    words_sum_pattern = r"words_sum:\s+(\d+)"
    bleu_pattern = r"bleu:\s+(\d+\.\d+)"
    log_cider_pattern = r"log_cider:\s+(\d+\.\d+)"

    rest_of_stdout = result.stdout.split("meteor")[-1]
    # print("rest_of_stdout", rest_of_stdout)
    # Extracting values using regex
    hallucination_sum_match = re.search(hallucination_sum_pattern, rest_of_stdout)
    object_sum_match = re.search(object_sum_pattern, rest_of_stdout)
    words_sum_match = re.search(words_sum_pattern, rest_of_stdout)
    bleu_match = re.search(bleu_pattern, rest_of_stdout)
    log_cider_match = re.search(log_cider_pattern, rest_of_stdout)

    print("log_cider_match", log_cider_match)


    # # Extracted values
    # hallucination_sum = int(hallucination_sum_match.group(1)) if hallucination_sum_match else None
    # metrics.append(str(hallucination_sum))
    # object_sum = int(object_sum_match.group(1)) if object_sum_match else None
    # metrics.append(str(object_sum))
    # words_sum = int(words_sum_match.group(1)) if words_sum_match else None
    # metrics.append(str(words_sum))
    bleu = float(bleu_match.group(1)) if bleu_match else None
    metrics.append(str(bleu)[:5])
    log_cider = float(log_cider_match.group(1)) if log_cider_match else None
    metrics.append(str(log_cider)[:5])

    # print("metrics", metrics)

    # Returning the extracted metrics
    return metrics


def extract_info_from_filename(filename):
    # Corrected regex pattern to accurately capture 'llava-1.5' as the model name
    match = re.search(
        r"([^_]+)_([^_]+)_([^_]+)_beams_(\d+)_k_(\d+)_coco_expand_ratio_([\d.]+)_seed_(\d+)_max_tokens_(\d+)_samples_(\d+)_chair",
        filename,
    )
    if match:
        # print(match.groups())
        return match.groups()
    else:
        return "-", "-", -1, -1, -1, -1, -1, -1


# Initialize the markdown table with headers
markdown_table = "| Backbone | Decoder | Detector | Ratio | Beam | K | Seed | SPICE | METEOR | CIDEr | CHAIRs | CHAIRi | BLEU | Log CIDEr |  Num of Samples | Max Tokens |\n"
markdown_table += "|---------|---------|---------|-----------|-----------|----------|------------|-------|--------|-------|-------|-------|--------|--------|--------|--------|\n"

# Prepare the CSV file
csv_file_path = "eval/eval_results.csv"
csv_columns = [
    "Backbone",
    "Decoder",
    "Detector",
    "Ratio",
    "Beam",
    "K",
    "Seed",
    "SPICE",
    "METEOR",
    "CIDEr",
    "CHAIRs",
    "CHAIRi",
    "BLEU",
    "Log CIDEr",
    "Max Tokens",
    "Num of Samples",
]

# Start writing to the CSV file
with open(csv_file_path, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()  # write the header

    file_names = os.listdir(directory)
    # Sort the file names by beam size, then by k number, then by seed number
    sorted_file_names = sorted(
        file_names, key=lambda name: extract_info_from_filename(name)[1:]
    )

    # print("file_names", file_names)
    # Loop through each file in the directory and process it
    for file_name in sorted_file_names:
        if file_name.endswith("_chair.json"):
            file_path = os.path.join(directory, file_name)
            print(file_path)
            # Extract information from filename
            # decoder,  beam_size, k_number, expand_ratio, seed_number = extract_info_from_filename(file_name)
            (
                backbone,
                decoder,
                detector,
                beam_size,
                k_number,
                expand_ratio,
                seed_number,
                max_tokens,
                num_samples,
            ) = extract_info_from_filename(file_name)

            metrics = run_eval(file_path)

            # print(f"| {decoder} | {expand_ratio} | {beam_size} | {k_number} | {seed_number} | {' | '.join(metrics)} |\n")
            print(
                f"| {backbone} | {decoder} | {detector} | {expand_ratio} | {beam_size} | {k_number} | {seed_number} | {' | '.join(metrics)} | {max_tokens} | {num_samples} |\n"
            )
            if metrics:
                # markdown_table += f"| {decoder} | {expand_ratio} | {beam_size} | {k_number} | {seed_number} | {' | '.join(metrics)} |\n"
                markdown_table += f"| {backbone} | {decoder} | {detector} | {expand_ratio} | {beam_size} | {k_number} | {seed_number} | {' | '.join(metrics)} | {max_tokens} | {num_samples} |\n"
                writer.writerow(
                    {
                        "Backbone": backbone,
                        "Decoder": decoder,
                        "Detector": detector,
                        "Ratio": expand_ratio,
                        "Beam": beam_size,
                        "K": k_number,
                        "Seed": seed_number,
                        "SPICE": metrics[0],
                        "METEOR": metrics[1],
                        "CIDEr": metrics[2],
                        "CHAIRs": metrics[3],
                        "CHAIRi": metrics[4],
                        "BLEU": metrics[5],
                        "Log CIDEr": metrics[6],
                        "Max Tokens": max_tokens,
                        "Num of Samples": num_samples,
                    }
                )


# Save the markdown table to a file
markdown_file_path = "eval/eval_results.md"
with open(markdown_file_path, "w") as md_file:
    md_file.write(markdown_table)

# The CSV file is already saved at this point, so we can output the paths to both files
print(f"Markdown file saved to: {markdown_file_path}")
print(f"CSV file saved to: {csv_file_path}")
