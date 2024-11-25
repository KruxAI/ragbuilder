csv_file_path = "/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/ll_ap.csv"
with open(csv_file_path, 'r') as f:
    for line_number, line in enumerate(f, start=1):
        if len(line.split(',')) != 29:  # Replace 29 with the expected number of fields
            print(f"Issue in line {line_number}: {line}")