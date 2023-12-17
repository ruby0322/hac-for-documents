
from pa4_cli import parse_args
from pa4_io import *
from pa4_preprocess import *
from pa4_process import *


def log(message: str):
    print(f'[IRTM PA4] {message}')


def main():
    try:
        log('starting...')
        args = parse_args()

        custom_stopwords = set()
        custom_contractions = {}
        if args.stopwords_file:
            log('loading custom stopwords file...')
            custom_stopwords = load_custom_stopwords(args.stopwords_file)
        if args.contractions_file:
            log('loading custom contractions file...')
            custom_contractions = load_custom_contractions(
                args.contractions_file)

        log('loading documents...')
        documents = load_documents(args.data_folder)

        log('preprocessing documents...')
        prep_documents = preprocess(documents, custom_stopwords, custom_contractions,
                                    args.remove_stopwords, args.expand_contractions, args.stem)

        log('building dictionary...')
        dictionary = build_dictionary(prep_documents)

        if args.export_dictionary:
            log('exporting dictionary...')
            export_dictionary(dictionary, args.output_folder)

        log('calculating TF-IDF vectors...')
        tfidf_matrix = create_tfidf_vectors(prep_documents, dictionary)

        if args.export_tfidf:
            log('exporting TF-IDF vectors...')
            export_tfidf_vectors(tfidf_matrix, args.output_folder)

        if args.hac:
            log('performing HAC...')
            A = hac(tfidf_matrix)

            if args.cluster:
                def run(K):
                    log(f'clustering documents into {K} clusters...')
                    clusters = form_k_clusters(A, K, len(documents))
                    write_clusters_to_file(clusters, K, args.output_folder)
                if ',' in args.cluster:
                    log(f'preparing to cluster documents into {args.cluster} clusters respectively...')
                    Ks = list(map(int, args.cluster.split(',')))
                    for K in Ks:
                        run(K)
                else:
                    log(f'preparing to cluster documents into {args.cluster} clusters...')
                    K = int(args.cluster)
                    run(K)

        return {
            'status_code': 0,
            'message': 'OK',
        }
    except Exception as e:

        return {
            'status_code': 1,
            'message': 'Error',
            'error': e.__str__()
        }


# Main Program
if __name__ == "__main__":
    res = main()
    log(
        f'program exited with status code {res["status_code"]} ({res["message"]})')
    if (res['status_code'] == 1):
        log(res['error'])
