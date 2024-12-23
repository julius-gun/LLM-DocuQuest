import os
import json
# Define the path to the JSON file
script_dir = os.path.dirname(__file__)
question_pairs_path = os.path.normpath(os.path.join(script_dir, '..', '..', 'question_answer_pairs.json'))
# Load the JSON data from the file
with open(question_pairs_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
# Sort the data by the 'page' field
sorted_data = sorted(data, key=lambda x: x['page'])
# Write the sorted data back to the file
with open(question_pairs_path, 'w', encoding='utf-8') as file:
    json.dump(sorted_data, file, indent=4, ensure_ascii=False)
print("JSON data sorted by page and saved successfully.")