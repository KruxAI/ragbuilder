import gensim.parsing.preprocessing as gpp
import nltk  # For example, if you want to add NLTK functions
# import spacy or other libraries if needed
import os
import logging
from multiprocessing import Pool
from tqdm import tqdm
from urllib.parse import urlparse
from pathlib import Path
import requests
from unstructured.partition.auto import partition
logger = logging.getLogger("ragbuilder")
# List of processor names, categorized by their library or origin

DATA_PROCESSORS = [
    "gpp:remove_stopwords",
    "gpp:strip_tags",
    "gpp:strip_punctuation",
    "gpp:strip_multiple_whitespaces",
    "gpp:strip_short",
    "gpp:stem_text",
]

# Function to resolve processors based on the prefix (library)
def resolve_processor(processor_name: str):
    # Split the processor into library and function name
    library, func_name = processor_name.split(":")
    if library == "gpp":
        return getattr(gpp, func_name)
    elif library == "custom": # For custom functions defined in the script
        return globals()[func_name]
    elif library == "nltk":
        return getattr(nltk, func_name)
    else:
        raise ValueError(f"Unknown library: {library}")

class DataProcessor:
    def __init__(self, data_source: str, data_processors: list = None):
        self.data_source = data_source
        # Resolve the processors using their library and function names
        self.data_processors = [resolve_processor(p) for p in (data_processors if data_processors is not None else DATA_PROCESSORS)]
        try:
            self.processed_data = self.apply_processors()
        except Exception as e:
            logger.error(f"Failed to process data: {str(e)}")
            self.processed_data = None

    def is_url(self, path: str) -> bool:
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except ValueError:
            logger.warning(f"Invalid URL: {path}")
            return False

    def apply_processors(self) -> str:
        if isinstance(self.data_source, str):
            path = Path(self.data_source)
            try:
                if self.is_url(self.data_source):
                    return self.process_url()
                elif path.is_file():
                    return self.process_file(str(path))
                elif path.is_dir():
                    return self.process_directory(str(path))
            except Exception as e:
                logger.error(f"Error processing {self.data_source}: {str(e)}")
                return None

        data = self.data_source
        for processor in self.data_processors:
            try:
                data = processor(data)
            except Exception as e:
                logger.warning(f"Processor {processor.__name__} failed: {str(e)}")
        return data

    def process_directory(self, dir_path: str) -> str:
        try:
            processed_dir = f"{dir_path}_processed"
            os.makedirs(processed_dir, exist_ok=True)

            files = [f for f in Path(dir_path).rglob('*') if f.is_file() and f.name != '.DS_Store']
            args = [(str(f), f'{processed_dir}/{str(f.relative_to(dir_path))}.processed') for f in files]
            # print(args)

            # Use the Pool within the main guard
            with Pool() as pool:
                list(tqdm(
                    pool.starmap(self.process_file, args),
                    total=len(files),
                    desc=f"Processing directory {dir_path}"
                ))

            return processed_dir
        except Exception as e:
            logger.error(f"Error processing directory {dir_path}: {str(e)}")
            return None

    def process_file(self, file_path: str, processed_file: str = None) -> str:
        try:
            if not processed_file:
                # Generate a processed file path based on the original file path
                processed_file = f"{file_path}.processed"

            # Ensure the directory for the processed file exists
            processed_dir = os.path.dirname(processed_file)
            if processed_dir:
                os.makedirs(processed_dir, exist_ok=True)

            # print(file_path, processed_file)
            logger.info(f"Preprocessing file: {file_path}")

            # Read the original file and process it
            with open(file_path, "rb") as f:
                elements = partition(file=f, include_page_breaks=True)
                file_content = "\n".join([str(el) for el in elements])

            # Process the content with the defined processors
            processed_content = file_content  # Start with the original content
            for processor in self.data_processors:
                processed_content = processor(processed_content)  # Update processed_content in each iteration

            # Write the processed content to the new file
            with open(processed_file, 'w', encoding='utf-8') as f:
                f.write(processed_content)

            logger.info(f"Preprocessing complete for {file_path}. Saved to {processed_file}")
            return processed_file
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return None

    def process_url(self) -> str:
        try:
            url = self.data_source
            response = requests.get(url)
            response.raise_for_status()

            # Save the content to a temporary file
            temp_file = f"temp_url_content_{hash(url)}"
            with open(temp_file, 'wb') as f:
                f.write(response.content)

            # Sample the temporary file
            processed_file = self.process_file(temp_file)

            # Remove the temporary file if it wasn't the one returned
            if processed_file != temp_file:
                os.remove(temp_file)
                logger.info(f"Processed URL content saved to: {processed_file}")

            return processed_file
        except Exception as e:
            logger.error(f"Error processing URL {self.data_source}: {str(e)}")
            return None

# if __name__ == "__main__":
#     # #directory
#     print("process dirs")
#     filename='/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/testfolder'
#     processor = DataProcessor(data_source=filename, data_processors=["gpp:remove_stopwords"])
#     print(processor.processed_data) 
#     # #file
#     print("process files")
#     filename='/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/testfile.txt'
#     processor = DataProcessor(data_source=filename, data_processors=["gpp:remove_stopwords"])
#     print(processor.processed_data) 


#     print("process urls")
#     url='https://ashwinaravind.github.io/'
#     processor = DataProcessor(data_source=url, data_processors=["gpp:remove_stopwords"])
#     print(processor.processed_data)  # Output: 'temp_url_content_123456.processed' (temporary file)


#     #file
#     print("unstructured files")
#     filename='/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/arxiv.pdf'
#     processor = DataProcessor(data_source=filename, data_processors=["gpp:remove_stopwords"])
#     print(processor.processed_data) 
