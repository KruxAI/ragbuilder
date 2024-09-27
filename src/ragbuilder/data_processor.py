import gensim.parsing.preprocessing as gpp
import nltk  # For example, if you want to add NLTK functions
# import spacy or other libraries if needed
import os
import logging
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
    # Add more processors from other libraries as needed
]

# Function to resolve processors based on the prefix (library)
def resolve_processor(processor_name: str):
    # Split the processor into library and function name
    library, func_name = processor_name.split(":")
    
    if library == "gpp":
        return getattr(gpp, func_name)
    elif library == "custom":
        return globals()[func_name]  # For custom functions defined in the script
    elif library == "nltk":
        return getattr(nltk, func_name)
    else:
        raise ValueError(f"Unknown library: {library}")

class DataProcessor:
    def __init__(self, data_source: str, data_processors: list = None):
        self.data_source = data_source
        # Resolve the processors using their library and function names
        self.data_processors = [resolve_processor(p) for p in (data_processors if data_processors is not None else DATA_PROCESSORS)]
        self.processed_data = self.apply_processors()

    def is_url(self, path: str) -> bool:
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def apply_processors(self) -> str:
        if isinstance(self.data_source, str):
            path = Path(self.data_source)
            if self.is_url(self.data_source):
                return self.process_url()
            elif path.is_file():
                return self.process_file(str(path))
            elif path.is_dir():
                return self.process_directory(str(path))
        data = self.data_source
        for processor in self.data_processors:
            data = processor(data)  # Apply each processor to the data
        return data

    def process_directory(self, dir_path: str) -> str:
        processed_dir = f"{dir_path}_processed"
        os.makedirs(processed_dir, exist_ok=True)

        # Get all files recursively, excluding '.DS_Store'
        files = [f for f in Path(dir_path).rglob('*') if f.is_file() and f.name != '.DS_Store']

        for f in files:
            # Get the relative path of the file to maintain folder structure in processed output
            relative_path = f.relative_to(dir_path)
            output_file_dir = Path(processed_dir) / relative_path.parent

            # Create subdirectories if necessary
            output_file_dir.mkdir(parents=True, exist_ok=True)

            # Process the file and save it in the corresponding subdirectory
            self.process_file(f, str(output_file_dir), filename=f.name)
        
        return processed_dir
    
    def process_file(self, file_path: str,process_dir: str=None,filename:str=None) -> str: 
        if process_dir is None:
            processed_file = f"{file_path}_processed"
        else:
            processed_file = f"{process_dir}/{filename}.processed"
        with open(file_path, "rb") as f:
            elements = partition(file=f, include_page_breaks=True)
            file_content="\n".join([str(el) for el in elements])
        for processor in self.data_processors:
            processed_content = processor(file_content)
        with open(processed_file, 'w', encoding='utf-8') as f:
            f.write(processed_content)
        return processed_file

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
                logger.info(f"Procssed URL content saved to: {processed_file}")

            return processed_file
        except Exception as e:
            logger.error(f"Error sampling URL {self.data_source}: {str(e)}")
            raise

# #directory
print("process dirs")
filename='/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/testfolder'
processor = DataProcessor(data_source=filename, data_processors=["gpp:remove_stopwords"])
print(processor.processed_data) 


# #file
# print("process files")
# filename='/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/testfile.txt'
# processor = DataProcessor(data_source=filename, data_processors=["gpp:remove_stopwords"])
# print(processor.processed_data) 


# # print("process urls")
# url='https://ashwinaravind.github.io/'
# processor = DataProcessor(data_source=url, data_processors=["gpp:remove_stopwords"])
# print(processor.processed_data)  # Output: 'temp_url_content_123456.processed' (temporary file)


# #file
# print("unstructured files")
# filename='/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/arxiv.pdf'
# processor = DataProcessor(data_source=filename, data_processors=["gpp:remove_stopwords"])
# print(processor.processed_data) 

