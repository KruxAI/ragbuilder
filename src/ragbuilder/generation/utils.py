from datasets import Dataset
import pandas as pd
from ragbuilder.generation.config import EvalDataset
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