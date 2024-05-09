import argparse
import pandas as pd
import os

def merge_datasets(input_dir: str, output_file: str):
  csv_files = [file for file in os.listdir(input_dir) if file.endswith('.csv')]

  # Initialize an empty DataFrame to store merged data
  merged_data = pd.DataFrame()

  # Iterate through each CSV file and merge its data into the DataFrame
  for file in csv_files:
      file_path = os.path.join(input_dir, file)
      df = pd.read_csv(file_path)
      merged_data = pd.concat([merged_data, df], ignore_index=True)

  # Write the merged data to the output CSV file
  merged_data.to_csv(output_file, index=False)

print("Merged CSV file created successfully!")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Merge CSV files')
    parser.add_argument('--input', '-I', help='Input directory with files to merge')
    parser.add_argument('--output', '-O', help='Output CSV file')
    args = parser.parse_args()

    # Check if input and output files are provided
    if not args.input or not args.output:
        parser.error('Please provide both input and output.')

    # Clean the CSV file
    merge_datasets(args.input, args.output)

if __name__ == '__main__':
    main()