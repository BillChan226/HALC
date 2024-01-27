import os
import json

def read_json_files(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json') and "halc" in filename:
            file_path = os.path.join(folder_path, filename)
            # with open(file_path, 'r') as file:
            #     file_data = json.load(file)
            file_data = []
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    pair = json.loads(line)
                    file_data.append(pair)
                length = len(file_data)
                data.extend(file_data)
    return data, length

def calculate_accumulative_score(data):
    aggregate_answers = {}
    for item in data:
        image_id = item['image_id']
        answer = item['answer']
        if image_id not in aggregate_answers:
            aggregate_answers[image_id] = []

        aggregate_answers[image_id].append(answer)

    return sum(any(answers) for answers in aggregate_answers.values())

# Replace 'your_folder_path' with the actual folder path containing JSON files
folder_path = '/home/czr/HaLC/log/llava-1.5'
combined_data, length = read_json_files(folder_path)
accumulative_score = calculate_accumulative_score(combined_data)

accumulative_rate = accumulative_score / length

print("Accumulative Score:", accumulative_score)
print("Accumulative Rate:", accumulative_rate)
