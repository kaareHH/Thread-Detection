import argparse
import pandas as pd
import numpy as np

def clean_csv(input_file: str, output_file: str):
  # Read the CSV file
  df = pd.read_csv(input_file)

  # Clean up column names by stripping whitespace
  df.columns = df.columns.str.strip()

  # Remove rows with missing values
  df = df.dropna()

  # Find rows containing 'Infinity' or 'inf' values
  rows_with_infinity = df[(df.eq('Infinity') | df.eq(np.inf)).any(axis=1)]

  # Remove rows with 'Infinity' or 'inf' values
  df = df.drop(rows_with_infinity.index)

  # Write the cleaned DataFrame to a new CSV file
  df.to_csv(output_file, index=False)

  print(f"Cleaned data written to '{output_file}'.")

def main():
  # Parse command-line arguments
  parser = argparse.ArgumentParser(description='Clean CSV file')
  parser.add_argument('--input', '-I', help='Input CSV file')
  parser.add_argument('--output', '-O', help='Output CSV file')
  args = parser.parse_args()

  # Check if input and output files are provided
  if not args.input or not args.output:
      parser.error('Please provide both input and output files.')

  # Clean the CSV file
  clean_csv(args.input, args.output)

if __name__ == '__main__':
  main()