import os
import re
import subprocess

# Set the directory where the chair.json files are located
directory = './paper_result/minigpt4/'

# Function to run the eval_hallucination command and parse the output
def run_eval(file_path):
    # Running the eval_hallucination command for the given file
    result = subprocess.run(['python', 'eval_hallucination.py', '--chair_input_path', file_path],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # print(result.stdout)
    # Using regex to extract the metrics from the command output
    # metrics = re.findall(r'SPICE\s*(\d+\.\d+)\s*METEOR\s*(\d+\.\d+)\s*CIDEr\s*(\d+\.\d+)\s*CHAIRs\s*(\d+\.\d+)\s*CHAIRi\s*(\d+\.\d+)', 
    #                      result.stdout)
    metrics = re.findall(r'\d+\.\d+', result.stdout.split('ground truth captions')[-1].split('CHAIR results')[0])
    # Returning the extracted metrics
    return metrics if metrics else None

# Function to extract the decoder, beam size, k number, and seed number from the filename
def extract_info_from_filename(filename):
    match = re.search(r'(halc-\w+)_beams_(\d+)_k_(\d+)_coco_seed_(\d+)', filename)
    if match:
        return match.groups()
    else:
        return ('-', '-', '-', '-')



# Initialize the markdown table with headers
markdown_table = "| Decoder | Beam | K | Seed | SPICE | METEOR | CIDEr | CHAIRs | CHAIRi |\n"
markdown_table += "|---------|-----------|----------|------------|-------|--------|-------|--------|--------|\n"

file_names = os.listdir(directory)
# Sort the file names by beam size, then by k number, then by seed number
sorted_file_names = sorted(file_names, key=lambda name: extract_info_from_filename(name)[1:])


# Loop through each file in the directory and process it
for file_name in sorted_file_names:
    if file_name.endswith('_chair.json'):
        file_path = os.path.join(directory, file_name)
        print(file_path)
        # Extract information from filename
        decoder, beam_size, k_number, seed_number = extract_info_from_filename(file_name)
        metrics = run_eval(file_path)
        print(f"| {decoder} | {beam_size} | {k_number} | {seed_number} | {' | '.join(metrics)} |\n")

        if metrics:
            markdown_table += f"| {decoder} | {beam_size} | {k_number} | {seed_number} | {' | '.join(metrics)} |\n"


# Save the markdown table to a file
markdown_file_path = 'eval/eval_results.md'
with open(markdown_file_path, 'w') as md_file:
    md_file.write(markdown_table)

markdown_file_path  # Return the path to the markdown file
