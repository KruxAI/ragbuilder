from datasets import Dataset
import pandas as pd
from ragbuilder.generation.config import EvalDataset
from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
def get_eval_dataset(eval_dataset_path) -> Dataset:
        """
        This function reads a CSV file, validates the required columns (`question` and `ground_truth`),
        converts each row to a Pydantic model, and returns a Dataset for Ragas compatibility.

        Args:
        - csv_file_path (str): Path to the CSV file.

        Returns:
        - Dataset: A Ragas-compatible dataset.

        Raises:
        - ValueError: If required columns (`question` and `ground_truth`) are missing or invalid.
        """
        # Load the CSV into a DataFrame
        df = pd.read_csv(eval_dataset_path)

        # Check if the required columns are present
        required_columns = ['question', 'ground_truth']
        missing_columns = [col for col in required_columns if col not in df.columns]
        df = df.dropna(subset=['ground_truth'])
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        # Convert the dataframe to a Pydantic model and validate
        try:
            eval_dataset_model = EvalDataset.from_dataframe(df)
        except ValueError as e:
            raise ValueError(f"Validation error: {e}")

        # Convert the validated Pydantic model to a dataset for Ragas
        result_dict = [item.dict() for item in eval_dataset_model.items]
        eval_dataset = Dataset.from_list(result_dict)
        return eval_dataset

from typing import List
from langchain_core.documents import Document

class dummyRetriever(BaseRetriever):
    """A toy retriever that contains the top k documents that contain the user query."""

    documents: List[Document]
    k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        matching_documents = []
        for document in self.documents:
            if len(matching_documents) >= self.k:
                return matching_documents
            if query.lower() in document.page_content.lower():
                matching_documents.append(document)
        return matching_documents

    @staticmethod
    def strings_to_documents(strings: List[str]) -> List[Document]:
        """Converts a list of strings into a list of Document objects."""
        return [Document(page_content=string) for string in strings]

# Example usage
# strings = [
#     "A young woman.",
#     """## Chunk 3: The Locket’s Secret

# Clara awoke to a sound—footsteps. Her eyes shot open. In the dim light, she saw a man standing at the door, silhouetted against the night. His green eyes gleamed in the shadows.

# “Who are you?” she stammered, clutching her locket.

# “I could ask you the same,” the man replied, stepping forward. “This is my family’s cabin. What are you doing here?”
# """
# ]
# documents = dummyRetriever.strings_to_documents(strings)
# retriever = dummyRetriever(documents=documents, k=3)
