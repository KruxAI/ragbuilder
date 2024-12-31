import argparse
from  ragbuilder import RAGBuilder,DataIngestOptionsConfig,RetrievalOptionsConfig,RetrievalOptionsConfig

def main():
    parser = argparse.ArgumentParser(description='Start the RAG server.')
    parser.add_argument('--input_source', type=str, required=True, help='Path to the input source.')
    # parser.add_argument('--test_dataset', type=str, default=None, help='Name of the test dataset.')
    args = parser.parse_args()

    builder = RAGBuilder.from_source_with_defaults(
        input_source=args.input_source,
        # test_dataset=args.test_dataset
    )
    builder.optimize()
    builder.serve()

if __name__ == "__main__":
    main()