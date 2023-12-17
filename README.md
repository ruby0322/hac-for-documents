---
title: 112-1 IRTM PA4
author: Kuan-Cheng, Ku; B10705016
date: Dec. 18th, 2023
colorlinks: yes
urlcolor: blue
---

# Hierarchical Agglomerative Clustering (HAC) Project

[GitHub Repo](https://github.com/ruby0322/hac-for-documents)

## Overview

This project implements a complete Hierarchical Agglomerative Clustering system for text data. It encompasses text preprocessing, TF-IDF vectorization, clustering, and output generation. The system is designed to process a collection of text documents, build a dictionary, compute TF-IDF vectors, and optionally perform HAC to cluster these documents.


## Requirements

- Python 3.x
- NumPy
- NLTK (for PorterStemmer)


## Project Structure

- `pa4.py`: The main script orchestrating the entire workflow.
- `pa4_cli.py`: Manages command-line argument parsing.
- `pa4_io.py`: Handles all input/output operations, such as loading documents and exporting results.
- `pa4_preprocess.py`: Performs text preprocessing, including tokenization, contraction expansion, and optional stemming.
- `pa4_process.py`: Contains core functionalities for building the dictionary, creating TF-IDF vectors, and implementing the HAC algorithm.

## Usage Instructions

1. **Prepare Your Data**: Ensure your text documents are placed in a specified input folder.
   
2. **Run the Script**: Execute the main script using `python pa4.py` followed by required arguments:
   - `data_folder`: Path to the folder containing input text documents.
   - `output_folder`: Path to the output folder for storing results.
   - `--stopwords_file` (optional): Path to a custom stopwords file.
   - `--contractions_file` (optional): Path to a custom contractions file.
   - `--cluster` (optional): Specify the number of clusters for HAC.
   - `--export_dictionary` (optional): Flag to export the dictionary.
   - `--export_tfidf` (optional): Flag to export TF-IDF vectors.
   - `--remove_stopwords` (optional): Enable stopwords removal.
   - `--expand_contractions` (optional): Enable contractions expansion.
   - `--stem` (optional): Enable stemming using the Porter algorithm.
   - `--hac` (optional): Perform Hierarchical Agglomerative Clustering.

3. **View Results**: After processing, the results (dictionary, TF-IDF vectors, clusters) will be available in the output folder.


### Example Command

```bash
python pa4.py data output --stopwords_file stopwords.txt --contractions_file contractions.json --export_dictionary --export_tfidf --remove_stopwords --expand_contractions --stem --hac --cluster 8,13,20
```

## Implementation Details

- **Text Preprocessing**: Includes tokenization, optional stopwords removal, contractions expansion, and stemming.
- **Dictionary Building**: Extracts a sorted set of unique terms from the preprocessed documents.
- **TF-IDF Calculation**: Computes Term Frequency-Inverse Document Frequency vectors for each document, normalized to unit vectors.
- **HAC Algorithm**: Implements HAC using a complete-link strategy, generating a merge history from which specified numbers of clusters can be formed.
- **Output Generation**: Exports the dictionary, TF-IDF vectors, and clustered documents as specified by the user.

### Core Functionality: HAC Algorithm

The `hac` function in `pa4_process.py` is a critical component of this project. It implements the Hierarchical Agglomerative Clustering algorithm using a complete-link strategy. This function takes the unit TF-IDF matrix of documents as input and outputs the history of cluster merges, which is essential for forming the final clusters.

#### How it Works

1. **Input**: The input to the function is a matrix of TF-IDF vectors for the documents, normalized to unit length.
2. **Cosine Similarity Matrix**: Initially, the cosine similarity between all pairs of documents is calculated. Since the input vectors are **unit length**, their **dot product directly gives the cosine similarity**.
3. **Heap for Efficient Similarity Retrieval**: A **min heap is used** to efficiently retrieve the pair of clusters with the maximum similarity (minimum distance). This approach significantly speeds up the process of finding the closest clusters to merge.
4. **Agglomerative Clustering Process**:
   - Iteratively, the algorithm merges the pair of clusters with the highest similarity.
   - After each merge, the similarity matrix is updated to reflect the new distances. In complete-link clustering, the similarity between a new merged cluster and other clusters is the minimum similarity of the merged clusters to those other clusters.
   - The merge history is recorded in a list `A`, where each element is a tuple indicating which clusters were merged.
5. **Termination**: The process continues until all documents are merged into a single cluster, resulting in a complete history of how clusters were merged.

#### Usage in the Project

- The `hac` function is invoked in `pa4.py` when the `--hac` flag is used.
- After HAC is performed, if the `--cluster` flag is specified, the program uses the merge history to form the desired number of clusters with `form_k_clusters` function from `pa4_process.py`.
- This clustering result can then be written to files, providing a clear representation of how documents are grouped together.
