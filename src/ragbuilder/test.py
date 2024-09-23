from ragbuilder.data_processor import DataProcessor, DATA_PROCESSORS
# Example usage
data_source = "This is an <b>EXAMPLE</b> string, with punctuation and stopwords!"
processor = DataProcessor(data_source,[DATA_PROCESSORS.remove_stopwords])
print(processor.processed_data)  # Processed string
processor = DataProcessor(data_source,[DATA_PROCESSORS.strip_tags])
print(processor.processed_data)  # Processed string
