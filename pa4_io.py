import json
import os


# Load custom stopwords from "stopwords.txt"
def load_custom_stopwords(stopwords_file):
    custom_stopwords = set()
    with open(stopwords_file, 'r', encoding='utf-8') as file:
        for line in file:
            custom_stopwords.add(line.strip())
    return custom_stopwords

# Load custom contraction-expansion rules from "contractions.json"


def load_custom_contractions(contractions_file):
    with open(contractions_file, 'r', encoding='utf-8') as file:
        contraction_rules = json.load(file)
    return contraction_rules


def load_documents(folder_path):
    documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                text = file.read().lower()  # Convert to lowercase
                documents.append(text)

    return documents


def export_dictionary(dictionary, output_folder):
    if output_folder:
        dict_file_path = os.path.join(output_folder, "dictionary.txt")
        with open(dict_file_path, 'w', encoding='utf-8') as dict_file:
            dict_file.write('\n'.join(dictionary))


def export_tfidf_vectors(tfidf_matrix, output_folder):
    if output_folder:
        for i, doc_vector in enumerate(tfidf_matrix):
            doc_id = i + 1
            filename = os.path.join(output_folder, f"{doc_id}.txt")
            with open(filename, 'w', encoding='utf-8') as doc_file:
                tfidf_values = ' '.join(map(str, doc_vector))
                doc_file.write(tfidf_values)


def write_clusters_to_file(clusters, K, output_folder="output"):
    """
    Write the clusters to a text file, with each cluster separated by an empty line.

    Parameters:
    clusters (list of sets): List of clusters to be written to the file.
    output_folder (str): Path to the output folder.
    """
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Write clusters to file
    file_path = os.path.join(output_folder,f'{K}.txt')
    with open(file_path, 'w') as file:
        for cluster in clusters:
            for doc_id in cluster:
                file.write(f"{doc_id}\n")
            file.write("\n")

    return file_path

