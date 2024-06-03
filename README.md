# Information Retrieval System

This repository contains the implementation of an Information Retrieval (IR) system that processes, analyzes, and retrieves relevant documents based on user queries using datasets from clinical trials and lifestyle forums. The project incorporates various services for text processing, vectorization using TF-IDF, document ranking, clustering, and evaluation of the search results.

## Table of Contents

1. [Datasets](#datasets)
2. [Project Structure](#project-structure)
3. [Services Overview](#services-overview)
4. [Execution Flow](#execution-flow)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Installation and Setup](#installation-and-setup)
7. [Usage](#usage)

## Datasets

### First Dataset: "clinicaltrials/2021/trec-ct-2021"
- **Documents:** 375,580
- **Queries:** 75
- **Query File:** Contains 35,832 records matching the query file

### Second Dataset: "lotte/lifestyle/dev/forum"
- **Documents:** 268,893
- **Queries:** 2,100
- **Query File:** Contains multiple records

## Project Structure

- **get_data_set.py**: Restructures the raw datasets for processing.
- **text_processing.py**: Contains functions for preprocessing text data.
- **Tfidf.py**: Implements TF-IDF vectorization and similarity calculations.
- **Ranking.py**: Ranks documents based on similarity scores.
- **search_service.py**: Orchestrates the search process and interfaces with the UI.
- **auto_complete.py**: Provides query suggestions and updates based on user feedback.
- **clustering.py, clusteringsearch.py**: Implements clustering-based search techniques.
- **evaluate.py**: Evaluates the search engine's performance using various metrics.
- **main.py**: Entry point of the application.
- **document.html, index.html**: HTML files for the web interface.

## Services Overview

### Text Processing Service
Handles text preprocessing tasks such as tokenization, case conversion, URL and punctuation removal, stopword removal, stemming, and lemmatization.

### TF-IDF Service
Transforms text documents into TF-IDF vectors, calculates similarity scores, and manages TF-IDF models.

### Ranking Service
Sorts documents based on their relevance to the query using similarity scores.

### Search Service
Integrates with the user interface to handle search requests and return ranked results.

### Auto-Complete Service
Provides search suggestions and updates dictionaries based on user interactions.

### Clustering Search Service
Uses K-Means clustering to group similar documents and improve search results.

### Evaluate Service
Evaluates the IR system using metrics like Mean Average Precision (MAP), Precision@10, Mean Recall, and MRR.

## Execution Flow

### Without Clustering
1. **UI Interaction**: User selects dataset and enters a query.
2. **Auto-Complete Service**: Provides query suggestions.
3. **Search Service**: 
   - Processes the query using Text Processing Service.
   - Converts the query to a TF-IDF vector.
   - Calculates cosine similarity between the query vector and document vectors.
   - Sorts documents using Ranking Service.
4. **UI**: Displays sorted search results.
5. **Feedback**: User feedback updates auto-complete suggestions.

### With Clustering
1. **UI Interaction**: User selects dataset and enters a query.
2. **Auto-Complete Service**: Provides query suggestions.
3. **Search Service**:
   - Processes the query using Text Processing Service.
   - Converts the query to a TF-IDF vector.
   - Determines the best-matching cluster using Clustering Search Service.
   - Calculates similarity within the cluster.
   - Sorts documents using Ranking Service.
4. **UI**: Displays sorted search results.
5. **Feedback**: User feedback updates auto-complete suggestions.

## Evaluation Metrics
The system is evaluated using the following metrics:
- **Mean Average Precision (MAP)**
- **Precision@10**
- **Mean Recall**
- **Mean Reciprocal Rank (MRR)**

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/QaysAlsary/IR.git
   cd IR
2. Install the required dependencies:
pip install -r requirements.txt
3. Download and prepare the datasets as per the project requirements.

## Usage

1. Run the main application:
python main.py
2. Access the web interface at http://localhost:5000 to interact with the IR system.
3. Evaluate the system:
- **Run the evaluation script to measure the performance of the IR system.**
python evaluate.py
