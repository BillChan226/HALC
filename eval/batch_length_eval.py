import os
import re
import subprocess
import csv
import argparse
import json

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
    result = subprocess.run(['python', 'eval_hallucination.py', '--chair_input_path', file_path],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # print(result.stdout)
    # input()
    # Using regex to extract the metrics from the command output
    # metrics = re.findall(r'SPICE\s*(\d+\.\d+)\s*METEOR\s*(\d+\.\d+)\s*CIDEr\s*(\d+\.\d+)\s*CHAIRs\s*(\d+\.\d+)\s*CHAIRi\s*(\d+\.\d+)', 
    #                      result.stdout)
    metrics = re.findall(r'\d+\.\d+', result.stdout.split('ground truth captions')[-1].split('CHAIR results')[0])
    # Returning the extracted metrics
    return metrics if metrics else None

def extract_info_from_filename(filename):
    # Corrected regex pattern to accurately capture 'llava-1.5' as the model name
    match = re.search(r'([^_]+)_([^_]+)_beams_(\d+)_k_(\d+)_coco_expand_ratio_([\d.]+)_seed_(\d+)_max_tokens_(\d+)_samples_(\d+)_chair', filename)
    if match:
        return match.groups()
    else:
        return '-', '-', -1, -1, -1, -1, -1, -1

# Initialize the markdown table with headers
markdown_table = "| Backbone | Decoder | Ratio | Beam | K | Seed | SPICE | METEOR | CIDEr | CHAIRs | CHAIRi | Num of Samples | Max Tokens |\n"
markdown_table += "|---------|---------|-----------|-----------|----------|------------|-------|--------|-------|--------|--------|--------|--------|\n"

# Prepare the CSV file
csv_file_path = 'eval/eval_results.csv'
csv_columns = ['Backbone', 'Decoder', 'Ratio', 'Beam', 'K', 'Seed', 'SPICE', 'METEOR', 'CIDEr', 'CHAIRs', 'CHAIRi', 'Max Tokens', 'Num of Samples']

# Start writing to the CSV file
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()  # write the header


    file_names = os.listdir(directory)
    # Sort the file names by beam size, then by k number, then by seed number
    sorted_file_names = sorted(file_names, key=lambda name: extract_info_from_filename(name)[1:])


    # Loop through each file in the directory and process it
    for file_name in sorted_file_names:
        if file_name.endswith('_generated_captions.json'):
            file_path = os.path.join(directory, file_name)
            print(file_path)
            # Extract information from filename
            # decoder,  beam_size, k_number, expand_ratio, seed_number = extract_info_from_filename(file_name)

            length_sum = 0
            # load file_name
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    json_line = json.loads(line)
                    length = len(json_line['caption'].split())
                    length_sum += length
            length_avg = length_sum / len(lines)
            print(length_avg)
