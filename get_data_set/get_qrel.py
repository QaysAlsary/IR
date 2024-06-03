import pathlib
import xml.etree.ElementTree as ET
import json
from collections import defaultdict

def parse_query_file(query_file_path):
    topics = {}
    tree = ET.parse(query_file_path)
    root = tree.getroot()
    for topic in root.findall('topic'):
        query_id = topic.get('number')
        text = topic.text.strip()
        topics[query_id] = text
    return topics

def parse_qrel_file(qrel_file_path):
    qrel_data = defaultdict(list)
    with open(qrel_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            query_id = parts[0]
            doc_id = parts[2]
            relevance = int(parts[3])
            qrel_data[query_id].append({
                "doc_id": doc_id,
                "relevance": relevance
            })
    return qrel_data

def convert_to_jsonl(query_file_path, qrel_file_path, output_file_path):
    topics = parse_query_file(query_file_path)
    qrel_data = parse_qrel_file(qrel_file_path)
    
    jsonl_data = []
    for query_id, text in topics.items():
        entry = {
            "query_id": query_id,
            "text": text,
            "docs": qrel_data.get(query_id, [])
        }
        jsonl_data.append(entry)
    
    with open(output_file_path, 'w') as jsonl_file:
        for entry in jsonl_data:
            try:
                json_line = json.dumps(entry)
                jsonl_file.write(json_line + '\n')
            except Exception as e:
                print(f"Error converting to JSON for query_id {entry['query_id']}: {e}")

# Example usage
cwd = pathlib.Path().cwd()

query_file_path = cwd / "files" / 'clinicaltrials' / "topics2021_2.xml"  # Replace with the actual path to your query file
qrel_file_path = cwd / "files" / 'clinicaltrials' / "qrels2021_2.txt"   # Replace with the actual path to your qrel file
output_file_path = cwd / "files" / 'clinicaltrials' / "qrel.jsonl"

convert_to_jsonl(query_file_path, qrel_file_path, output_file_path)
print(f"JSONL file created at {output_file_path}")
