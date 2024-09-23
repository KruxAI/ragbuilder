# Import functions from different libraries (e.g., Gensim, custom, etc.)
from gensim.parsing.preprocessing import (
    strip_tags,
    strip_punctuation,
    strip_multiple_whitespaces,
    remove_stopwords,
    strip_short,
    stem_text
)

# Example of adding more processing functions from other libraries
def custom_normalizer(text: str) -> str:
    return text.lower()

# Define constants for the processing functions
DATA_PROCESSORS = [
    strip_tags,
    strip_punctuation,
    strip_multiple_whitespaces,
    remove_stopwords,
    strip_short,
    stem_text,
    custom_normalizer  # Example custom function
]

class DataProcessor:
    def __init__(self, data_source: str, data_processors: list = None):
        self.data_source = data_source
        # Set the default processors to the constant list, or use user-provided processors
        self.data_processors = data_processors if data_processors is not None else DATA_PROCESSORS
        self.processed_data = self.apply_processors()

    def apply_processors(self) -> str:
        data = self.data_source
        for processor in self.data_processors:
            data = processor(data)  # Apply each processor to the data
        return data

# Example usage
data_source = "This is an <b>EXAMPLE</b> string, with punctuation and stopwords!"
processor = DataProcessor(data_source,[remove_stopwords])
print(processor.processed_data)  # Processed string
processor = DataProcessor(data_source,[strip_tags])
print(processor.processed_data)  # Processed string
