import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import WordNetLemmatizer
import string
from nltk.stem import PorterStemmer
import re
from nltk.corpus import wordnet
from nltk import pos_tag

class TextProcessor:
    def __init__(self):
        # Initialize NLTK resources
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.punctuation_table = str.maketrans('', '', string.punctuation)
        self.pos_type_map = {'J': 'a', 'N': 'n', 'V': 'v', 'R': 'r'}


    def remove_phonetic_notation(self, text):
        # Assuming phonetic notation is enclosed within square brackets
        return re.sub(r'\[.*?\]', '', text)

    def tokenize_text(self, text):
        return word_tokenize(text)

    def convert_to_lowercase(self, tokens):
        return [token.lower() for token in tokens]

    def remove_urls(self, tokens):
        return [token for token in tokens if not token.startswith(('http://', 'https://'))]

    def remove_punctuation(self, tokens):
        return [token.translate(self.punctuation_table) for token in tokens]

    def process_and_standardize_dates_brackets(self, tokens):
        # Add your logic for processing and standardizing dates within brackets here
        return tokens

    def remove_stopwords(self, tokens):
        return [token for token in tokens if token not in self.stop_words]

    def stem_tokens(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]

    def join_tokens(self, tokens):
        return ' '.join(tokens)
    def lemmatize_tokens(self, tokens):
        pos_tags = pos_tag(tokens)
        return [self.lemmatizer.lemmatize(token, self.get_wordnet_pos(pos)) for token, pos in pos_tags]
    def get_wordnet_pos(self, treebank_tag):
        # Map POS tag to first character lemmatize() accepts
        return self.pos_type_map.get(treebank_tag[0], wordnet.NOUN)

    def process_text(self, text):
        text_without_phonetic = self.remove_phonetic_notation(text)
        tokens = self.tokenize_text(text_without_phonetic)
        tokens = self.convert_to_lowercase(tokens)
        tokens = self.remove_urls(tokens)
        tokens = self.remove_punctuation(tokens)
        tokens = self.process_and_standardize_dates_brackets(tokens)
        tokens = self.remove_stopwords(tokens)
        tokens = self.stem_tokens(tokens)
        tokens = self.lemmatize_tokens(tokens)
        return   tokens

    def process_generic(self, query):
        query = query.lower()
        tokens = word_tokenize(query)

        filtered_tokens = [
            token.translate(self.punctuation_table)
            for token in tokens
            if token not in self.stop_words
        ]
        filtered_tokens = [token for token in filtered_tokens if token]

        pos_tokens = nltk.pos_tag(filtered_tokens)

        lemmatized_tokens = [
            self.stemmer.stem(self.lemmatizer.lemmatize(word, self.pos_type_map.get(pos[0], 'n')))
            for word, pos in pos_tokens
        ]

        return lemmatized_tokens

    def process_specific(self, query):
        query = query.lower()
        tokens = word_tokenize(query)

        filtered_tokens = [
            token.translate(self.punctuation_table)
            for token in tokens
            if token not in self.stop_words
        ]
        filtered_tokens = [token for token in filtered_tokens if token]

        pos_tokens = nltk.pos_tag(filtered_tokens)

        lemmatized_tokens = [
            self.lemmatizer.lemmatize(word, self.pos_type_map.get(pos[0], 'n'))
            for word, pos in pos_tokens
        ]

        return lemmatized_tokens

    def process_json_data(self, input_file_url, process_type, output_file_url):
        data = self.load_from_json(input_file_url)
        if not isinstance(data, dict):
            print("Invalid JSON format: Data is not a dictionary.")
            return

        processed_data = {}
        for key, value in data.items():
            print(f"Processing document ID: {key}")
            if process_type == "generic":
                processed_data[key] = self.process_text(value)
            elif process_type == "specific":
                processed_data[key] = self.process_text(value)
            else:
                raise ValueError("Invalid process type.")

        self.write_to_json(processed_data, output_file_url)

    def write_to_json(self, processed_data, output_file):
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(processed_data, file, indent=4)

    def load_from_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
