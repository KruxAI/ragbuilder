import pandas as pd

# Load the CSV file uploaded by the user
file_path = '/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/test.csv'
# Read the file line by line into the DataFrame
import csv
def test2():
    df_line_by_line = []
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                # Convert each row into a DataFrame row and append it
                df_line_by_line.append(row)
                if i >= 5:  # Preview first 5 lines (can be adjusted)
                    break  # Limiting to first 5 rows for preview
        # Create a DataFrame from the list of rows
        df_line_by_line_df = pd.DataFrame(df_line_by_line)
        print(df_line_by_line_df)
    except Exception as e:
        str(e)  # Capture any error message if occurs
test2()