import json
import pathlib

def extract_keys(json_data):
    keys = list(json_data.keys())
    return keys

def save_to_json(keys, output_file):
    try:
        with open(output_file, 'w') as file:
            json.dump(keys, file, indent=4)
            print(f"Keys saved to '{output_file}' successfully.")
    except IOError:
        print(f"Error occurred while writing to '{output_file}'.")

cwd = pathlib.Path().cwd()

file_path = cwd / 'files' / 'lifestyle' / 'corpus.json'
# Load the JSON data from the file with UTF-8 encoding
with open(file_path, 'r', encoding='utf-8') as file:
    corpus_data = json.load(file)

# Extract keys from the JSON data
corpus_keys = extract_keys(corpus_data)

# Path to the file where you want to save the keys
output_file_path = 'C:/Users/ASUS/Desktop/5th year/فصل تاني/project/files/lifestyle/corpuskeys.json'

# Saving the keys to another JSON file
save_to_json(corpus_keys, output_file_path)
