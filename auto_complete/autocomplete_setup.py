import jsonlines
import json
from collections import defaultdict
import pathlib

# Path to the qrel JSONL file
cwd = pathlib.Path().cwd()
# Path to the qrel JSONL file
# qrel_file_path = cwd / "files" / 'lifestyle' / "qas.search.jsonl"
# autocomplete_dict_path = cwd / "files" / 'lifestyle' / "autocomplete_dict.json"
# query_logs_file_path = cwd / "files" / 'lifestyle' / "query_logs.json"
# query_frequencies_path = cwd / "files" / 'lifestyle' / "query_frequencies.json"
qrel_file_path = cwd / "files" / 'clinicaltrials' / "qrel.jsonl"
autocomplete_dict_path = cwd / "files" / 'clinicaltrials' / "autocomplete_dict.json"
query_logs_file_path = cwd / "files" / 'clinicaltrials' / "query_logs.json"
query_frequencies_path = cwd / "files" / 'clinicaltrials' / "query_frequencies.json"

# Initialize an empty dictionary to store term frequencies for autocomplete
autocomplete_dict = defaultdict(int)
# Initialize an empty list to store query logs
query_logs = []
# Initialize an empty dictionary to store query frequencies
query_frequencies = defaultdict(int)

# Read the qrel JSONL file and add terms to the autocomplete dictionary
with jsonlines.open(qrel_file_path) as reader:
    for obj in reader:
        query = obj.get("text", "").strip()
        if query:
            query_logs.append(query)
            query_frequencies[query] += 1
            terms = query.split()
            for term in terms:
                autocomplete_dict[term] += 1

# Save the autocomplete dictionary to a JSON file
with open(autocomplete_dict_path, 'w') as f:
    json.dump(autocomplete_dict, f)

# Save the query logs to a JSON file
with open(query_logs_file_path, 'w') as f:
    json.dump(query_logs, f)

# Save the query frequencies to a JSON file
with open(query_frequencies_path, 'w') as f:
    json.dump(query_frequencies, f)
