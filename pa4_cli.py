import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Text Processing and TF-IDF Vectorization")

    # Input and Output Folders (Default values are the same as before)
    parser.add_argument(
        "data_folder", help="Path to the folder containing input text documents")
    parser.add_argument("output_folder", default="output",
                        help="Path to the output folder (optional)")

    # Custom Files (Default values are None)
    parser.add_argument("--stopwords_file", default=None,
                        help="Path to custom stopwords file (optional)")
    parser.add_argument("--contractions_file", default=None,
                        help="Path to custom contractions file (optional)")
    parser.add_argument("--cluster", default=None,
                        help="Cluster the documents into specified number of clusters")

    # Export Options
    parser.add_argument("--export_dictionary", action="store_true",
                        help="Export the dictionary to a file")
    parser.add_argument("--export_tfidf", action="store_true",
                        help="Export TF-IDF vectors to files")

    # Options to Remove Stopwords and Expand Contractions
    parser.add_argument("--remove_stopwords", action="store_true",
                        help="Remove stopwords during preprocessing")
    parser.add_argument("--expand_contractions", action="store_true",
                        help="Expand contractions during preprocessing")
    parser.add_argument("--stem", action="store_true",
                        help="Stem with the Porter algorithm")
    parser.add_argument("--hac", action="store_true",
                        help="Perform HAC on the documents")

    return parser.parse_args()
