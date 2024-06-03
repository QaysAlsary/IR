import os
import json
import xml.etree.ElementTree as ET
import pathlib

def convert_file(input_file, output_file, file_type):
    output_path = pathlib.Path(output_file)
    if file_type == 1:
        return tsv_to_json(input_file, output_path)
    elif file_type == 2:
        return process_clinical_trial_xml(input_file, output_path)
    else:
        raise ValueError("Invalid file type. Please provide 1 or 2.")

def tsv_to_json(tsv_file_path, output_file_path):
    main_corpus = {}

    with open(tsv_file_path, 'r', encoding='utf-8') as collection_file:
        for line in collection_file:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            doc_id = parts[0]
            text = parts[1]
            main_corpus[doc_id] = text

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with output_file_path.open('w', encoding="utf-8") as corpusfile:
        json.dump(main_corpus, corpusfile, ensure_ascii=False, indent=4)

    return output_file_path

def process_clinical_trial_xml(input_folder, output_file_path):
    clinical_trials_data = {}

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".xml"):
                xml_file_path = os.path.join(root, file)
                with open(xml_file_path, 'r', encoding='utf-8') as f:
                    tree = ET.parse(f)
                    xml_root = tree.getroot()

                    doc_id = xml_root.find("id_info/nct_id").text
                    title = xml_root.find("brief_title").text if xml_root.find("brief_title") is not None else ""
                    condition = xml_root.find("condition").text if xml_root.find("condition") is not None else ""
                    summary = xml_root.find("brief_summary/textblock").text if xml_root.find("brief_summary/textblock") is not None else ""
                    detailed_description = xml_root.find("detailed_description/textblock").text if xml_root.find("detailed_description/textblock") is not None else ""
                    eligibility = xml_root.find("eligibility/criteria/textblock").text if xml_root.find("eligibility/criteria/textblock") is not None else ""

                    text = (f"{title} {condition} "
                            f"{summary} {detailed_description} "
                            f"{eligibility}")  # Include eligibility here

                    # Replacing multiple spaces and ensuring the text follows the structure you mentioned
                    text = ' '.join(text.split())
                    text = text.replace("\n ", "\n")

                    clinical_trials_data[doc_id] = text

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with output_file_path.open('w', encoding="utf-8") as json_file:
        json.dump(clinical_trials_data, json_file, ensure_ascii=False, indent=4)

    return output_file_path
