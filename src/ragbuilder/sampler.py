import os
import random
import logging
import shutil
from typing import Tuple, List, TYPE_CHECKING, Any
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import requests
from urllib.parse import urlparse
# from ragbuilder.langchain_module.common import setup_logging
# from ragbuilder.analytics import track_event

if TYPE_CHECKING:
    from unstructured.partition.auto import partition
    from unstructured.documents.elements import Element

# setup_logging()
logger = logging.getLogger("ragbuilder.sampler")
SAMPLING_RATIO = float(os.getenv('SAMPLING_RATIO', '0.1'))
SAMPLING_SIZE_THRESHOLD = int(os.getenv('SAMPLING_SIZE_THRESHOLD', '750_000'))
SAMPLING_FILE_SIZE_THRESHOLD = int(os.getenv('SAMPLING_FILE_SIZE_THRESHOLD', '500_000'))


class DataSampler:
    def __init__(
            self, 
            data_source: str, 
            enable_sampling: bool = True, 
            sample_size_threshold: int =  SAMPLING_SIZE_THRESHOLD,
            sample_ratio: float = SAMPLING_RATIO, 
            file_size_threshold: int = SAMPLING_FILE_SIZE_THRESHOLD
    ):
        self.data_source = data_source
        self.enable_sampling = enable_sampling
        self.sample_size_threshold = sample_size_threshold
        self.sample_ratio = sample_ratio
        self.file_size_threshold = file_size_threshold
        self.random_state = 42
        # self.logger = self._setup_logger()

    # def _setup_logger(self):
    #     logger = logging.getLogger(__name__)
    #     logger.setLevel(logging.INFO)
    #     handler = logging.StreamHandler()
    #     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #     handler.setFormatter(formatter)
    #     logger.addHandler(handler)
    #     return logger

    def is_url(self, path: str) -> bool:
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def estimate_data_size(self) -> int:
        if isinstance(self.data_source, str):
            if self.is_url(self.data_source):
                return self.estimate_url_size(self.data_source)
            path = Path(self.data_source)
            if path.is_file():
                return path.stat().st_size
            elif path.is_dir():
                return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file() and str(f.relative_to(path)) != '.DS_Store')
        raise ValueError("Unsupported data source type")

    def estimate_url_size(self, url: str) -> int:
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36'}
            response = requests.head(url, headers=headers, allow_redirects=True)
            content_length = response.headers.get('Content-Length')
            if content_length:
                return int(content_length)
            else:
                response = requests.get(url, stream=True)
                return int(response.headers.get('Content-Length', 0))
        except Exception as e:
            logger.error(f"Error estimating URL size: {str(e)}")
            raise

    def need_sampling(self) -> bool:
        return self.estimate_data_size() > self.sample_size_threshold

    def sample_data(self) -> str:
        try:
            if not (self.need_sampling() and self.enable_sampling):
                logger.info(f"Sampling not required for {self.data_source}")
                return self.data_source

            if isinstance(self.data_source, str):
                if self.is_url(self.data_source):
                    # TODO: Think about downstream when sampling URL. For now, skip sampling URL sources
                    return self.data_source
                    # return self.sample_url()
                path = Path(self.data_source)
                if path.is_file():
                    return self.sample_file(str(path))
                elif path.is_dir():
                    return self.sample_directory(str(path))
                
            raise ValueError("Unsupported data source type")
        except Exception as e:
            logger.error(f"Error sampling data: {str(e)}")
            raise

    def sample_file(self, file_path: str, sampled_file_path: str = None) -> str:
        try:
            if Path(file_path).stat().st_size <= self.file_size_threshold:
                logger.info(f"File {file_path} is small, no need for content sampling")
                return file_path

            if not self.enable_sampling:
                return file_path

            # Only import when sampling is enabled
            from unstructured.partition.auto import partition
            from unstructured.documents.elements import Element

            logger.info(f"Sampling file: {file_path} (this may take a while)...")
            elements = partition(filename=file_path)
            total_elements = len(elements)
            sample_size = int(total_elements * self.sample_ratio)
            
            if sample_size == 0 or sample_size == total_elements:
                logger.info(f"File {file_path} doesn't need sampling")
                return file_path
            
            window_size = min(10, max(1, int(total_elements * 0.01)))  # 1% of total elements, min 1, max 10
            num_windows = int(total_elements * self.sample_ratio / window_size)
            sampled_elements = self._sliding_window_sample(elements, window_size, num_windows)

            # random.seed(self.random_state)
            # sample_indices = sorted(random.sample(range(total_elements), sample_size))
            # sampled_elements = [elements[i] for i in sample_indices]
            
            if not sampled_file_path:
                sampled_file_path = f"{file_path}.sampled"
            with open(sampled_file_path, 'w', encoding='utf-8') as f:
                for element in sampled_elements:
                    f.write(str(element) + '\n')
            
            logger.info(f"Sampled {len(sampled_elements)} elements from {file_path}. Saved to {sampled_file_path}")
            return sampled_file_path
        except Exception as e:
            logger.error(f"Error sampling file {file_path}: {str(e)}")
            raise

    def _sliding_window_sample(self, elements: List[Any], window_size: int, num_windows: int) -> List[Any]:
        total_elements = len(elements)
        max_start_index = total_elements - window_size

        if max_start_index <= 0:
            return elements  # If the document is shorter than the window size, return all elements

        random.seed(self.random_state)
        start_indices = sorted(random.sample(range(max_start_index + 1), min(num_windows, max_start_index + 1)))
        
        sampled_elements = []
        for start_index in start_indices:
            sampled_elements.extend(elements[start_index:start_index + window_size])
        
        return sampled_elements

    def analyze_directory(self, dir_path: str) -> Tuple[int, int, float]:
        total_files = 0
        total_size = 0
        for f in Path(dir_path).glob('**/*'):
            if f.is_file():
                total_files += 1
                total_size += f.stat().st_size
        avg_file_size = total_size / total_files if total_files > 0 else 0
        return total_files, total_size, avg_file_size

    def sample_directory(self, dir_path: str) -> str:
        try:
            total_files, total_size, avg_file_size = self.analyze_directory(dir_path)
            sampled_dir = f"{dir_path}_sampled"
            os.makedirs(sampled_dir, exist_ok=True)

            if avg_file_size > self.file_size_threshold:
                return self.file_level_sampling(dir_path, sampled_dir)
            else:
                return self.directory_level_sampling(dir_path, sampled_dir, total_files)
        except Exception as e:
            logger.error(f"Error sampling directory {dir_path}: {str(e)}")
            raise

    def file_level_sampling(self, dir_path: str, sampled_dir: str) -> str:
        files = [f for f in Path(dir_path).glob('**/*') if f.is_file() and str(f.relative_to(dir_path)) != '.DS_Store']
        args = [(str(f), f'{sampled_dir}/{str(f.relative_to(dir_path))}.sampled') for f in files]
        
        with Pool() as pool:
            results = list(tqdm(
                pool.starmap(self.sample_file, args),
                total=len(files),
                desc=f"Sampling directory {dir_path} (file-level)"
            ))
        
        logger.info(f"Sampled directory saved to: {sampled_dir}")
        return sampled_dir

    def directory_level_sampling(self, dir_path: str, sampled_dir: str, total_files: int) -> str:
        target_files = int(total_files * self.sample_ratio)
        files = list(Path(dir_path).glob('**/*'))
        random.seed(self.random_state)
        sampled_files = random.sample([f for f in files if f.is_file() and str(f.relative_to(dir_path)) != '.DS_Store'], target_files)

        for file in tqdm(sampled_files, desc=f"Sampling directory {dir_path} (directory-level)"):
            relative_path = file.relative_to(dir_path)
            target_path = Path(sampled_dir) / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            if file.stat().st_size > self.file_size_threshold:
                self.sample_file(str(file), target_path)
            else:
                shutil.copy2(file, target_path)

        logger.info(f"Sampled directory saved to: {sampled_dir}")
        return sampled_dir

    def sample_url(self) -> str:
        try:
            url = self.data_source
            response = requests.get(url)
            response.raise_for_status()
            
            # Save the content to a temporary file
            temp_file = f"temp_url_content_{hash(url)}"
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            
            # Sample the temporary file
            sampled_file = self.sample_file(temp_file)
            
            # Remove the temporary file if it wasn't the one returned
            if sampled_file != temp_file:
                os.remove(temp_file)
                logger.info(f"Sampled URL content saved to: {sampled_file}")

            return sampled_file
        except Exception as e:
            logger.error(f"Error sampling URL {self.data_source}: {str(e)}")
            raise

