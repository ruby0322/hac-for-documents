import math
from collections import Counter
import heapq
import numpy as np


def build_dictionary(documents):
    all_terms = [term for doc_tokens in documents for term in doc_tokens]
    dictionary = sorted(set(all_terms))
    return dictionary


def create_tfidf_vectors(documents, dictionary):
    # Calculate the Term Frequencies (TF) for each document
    tf_matrix = np.zeros((len(documents), len(dictionary)), dtype=float)

    for i, doc_tokens in enumerate(documents):
        term_freqs = Counter(doc_tokens)
        for j, term in enumerate(dictionary):
            tf_matrix[i, j] = term_freqs.get(term, 0)

    # Calculate the Inverse Document Frequencies (IDF) for each term
    num_documents = len(documents)
    doc_count = np.zeros(len(dictionary), dtype=float)

    for j, term in enumerate(dictionary):
        doc_count[j] = sum(1 for doc in documents if term in doc)

    idf_vector = 1 + np.log((1 + num_documents) / (1 + doc_count))

    # Calculate the TF-IDF vectors
    tfidf_matrix = tf_matrix * idf_vector

    # Normalize the TF-IDF vectors to unit vectors
    norms = np.linalg.norm(tfidf_matrix, axis=1)
    tfidf_matrix = tfidf_matrix / norms[:, np.newaxis]

    return tfidf_matrix


def sim(j, i, m, similarity_matrix):
    """
    Compute the new similarity between cluster 'j' and the merged cluster 'i' and 'm'.
    For simplicity, we'll use the average linkage method.
    """
    return min(similarity_matrix[j][i], similarity_matrix[j][m])

def hac(unit_tfidf_matrix):
    N = len(unit_tfidf_matrix)
    # Since vectors are unit vectors, the dot product directly gives the cosine similarity
    similarity_matrix = np.dot(unit_tfidf_matrix, unit_tfidf_matrix.T)
    I = np.ones(N)
    A = []

    heap = []
    for i in range(N):
        for j in range(i + 1, N):
            # Using negative similarity because heapq is a min heap, but we need max similarity
            heapq.heappush(heap, (-similarity_matrix[i][j], i, j))

    for _ in range(N - 1):
        while True:
            sim_value, i, m = heapq.heappop(heap)
            if I[i] == 1 and I[m] == 1:
                break

        A.append((i, m))

        # Update similarity_matrix for complete link
        for j in range(N):
            if j != i and I[j] == 1:
                new_sim = sim(j, i, m, similarity_matrix)
                similarity_matrix[i][j] = similarity_matrix[j][i] = new_sim
                heapq.heappush(heap, (-new_sim, j, i))
        I[m] = 0

    return A

def form_k_clusters(A, K, N):
    """
    Form K clusters from the merge history A.

    Parameters:
    A (list of tuples): Merge history from HAC, where each tuple represents a merge operation.
    K (int): Number of desired clusters.
    N (int): Total number of original documents.

    Returns:
    list of sets: List containing K clusters.
    """
    # Initialize clusters with each document as a separate cluster
    clusters = [{i} for i in range(N)]

    # Merge clusters as per the history in A
    for i in range(N - K):
        merge_i, merge_j = A[i]
        clusters[merge_i] = clusters[merge_i].union(clusters[merge_j])
        clusters[merge_j] = set()

    # Remove empty clusters and return only K clusters
    return [cluster for cluster in clusters if cluster]
