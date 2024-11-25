import pandas as pd

# Read the source CSV file
source_csv_file = '/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/rag_eval_results.csv'  # Replace with your actual file name
results_df = pd.read_csv(source_csv_file)

# Group by 'prompt_key' and calculate the mean of 'answer_correctness'
average_correctness = results_df.groupby('promt_key')['answer_correctness'].mean().reset_index()

# Rename columns for clarity (optional)
average_correctness.columns = ['promt_key', 'answer_correctness']

# Save the result to a new CSV file
output_csv_file = 'average_correctness.csv'  # Replace with desired output file name
average_correctness.to_csv(output_csv_file, index=False)

print(f"The results have been saved to '{output_csv_file}'")
