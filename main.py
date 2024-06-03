from get_data_set.get_data_set import convert_file
from pathlib import Path
import pathlib
from text_processing.TextProcessor import TextProcessor
from Tfidf.Tf_idf_Service import TfidfService
from invertedIndex.InvertedIndexService import InvertedIndexService
def main():
    # Example usage:
    input_file = 'C:/Users/ASUS/Desktop/5th year/فصل تاني/IR/dataset/ClinicalTrials'
    output_file = Path('C:/Users/ASUS/Desktop/5th year/فصل تاني/IR/project/project/files/clinicaltrials/corpus.json')
    clinical_folder ='C:/Users/ASUS/Desktop/5th year/فصل تاني/IR/dataset/ClinicalTrials' 

  
    # For TSV to JSON conversion
    #convert_file(input_file, output_file, 1)
    # For processing clinical trial XMLs
    # convert_file(clinical_folder, output_file, 2)
    ################################################################################
    ############################# Process data ######################################
    ################################################################################
    # Process JSON file
    # input_file = Path('C:/Users/ASUS/Desktop/5th year/فصل تاني/IR/project/project/files/clinicaltrials/corpus.json')
    # output_file = Path('C:/Users/ASUS/Desktop/5th year/فصل تاني/IR/project/project/files/clinicaltrials/corpus1.json')
    # processed_data = processor.process_json_file(input_file)
    # processor.write_to_json(processed_data, output_file)
    # Process a single query
    # query = "This is a sample query to be processed."
    # processed_query = processor.process(query)
    # print("Processed Query:", processed_query)
    ################################################################################
    ############################# TfidfService ######################################
    ################################################################################
    # service = TfidfService()
    # # Define file paths
    # cwd = pathlib.Path().cwd()
    # input_file_path = cwd /"files" /'clinicaltrials'/ "corpus1.json"
    # tfidf_matrix_path = cwd / "files"/'clinicaltrials' / "tfidf_matrix.pkl"
    # vectorizer_path = cwd /"files" /'clinicaltrials'/ "vectorizer.pkl"
    
    # # Process JSON file
    # service.process_json_file(input_file_path, tfidf_matrix_path, vectorizer_path)
    # print("TF-IDF processing of JSON file complete.")
    
    # # Process a single query
    # query_tokens = ["This", "is", "a", "sample", "query", "to", "be", "vectorized"]
    # processed_query = service.vectorize_query(query_tokens, vectorizer_path)
    # print("Processed Query TF-IDF:", processed_query)
    ################################################################################
    ############################ inverted index ######################################
    ################################################################################
   # Load pickled vectorizer and TF-IDF matrix
    # vectorizer_file = 'vectorizer.pkl'
    # tfidf_matrix_file = 'tfidf_matrix.pkl'
    # vectorizer, tfidf_matrix = InvertedIndexService.load_pickled_objects(vectorizer_file, tfidf_matrix_file)

    # # Initialize the TFIDF service with the loaded objects
    # tfidf_service = InvertedIndexService(vectorizer, tfidf_matrix)

    # # Create and write the inverted index to a file
    # tfidf_service.offline_write_index('inverted_index.json')

    # # Read the inverted index from the file
    # inverted_index = InvertedIndexService.offline_read_index('inverted_index.json')

    # print(inverted_index)

       ################################################################################
    ############################ inverted index ######################################
    ################################################################################


# python text_processing\TextProcessorservices.py


# python Tfidf\tfidf_server.py


# python similarity_service.py

# python Ranking\ranking_service.py
# python searchService\searchService.py



if __name__ == "__main__":
    main()
