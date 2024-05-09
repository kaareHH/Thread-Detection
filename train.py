from typing import Optional
import joblib
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(X_train, y_train)
    
    return rf_model


def continue_training(X_train, y_train, folder_path, existing_model):
    rf_model = joblib.load(os.path.join(folder_path, existing_model))
    rf_model.fit(X_train, y_train)
    
    return rf_model


def run(input_file: str, output_dir: str, show_plot: bool, existing_model: Optional[str]):
  folder_path = os.path.join(os.getcwd(), output_dir)

  if not os.path.exists(folder_path):
      os.makedirs(folder_path)

  df = pd.read_csv(input_file)

  # Map labels to numeric values
  # Get unique labels
  unique_labels = df['Label'].unique()

  # Map unique labels to numeric values
  label_mapping = {label: index for index, label in enumerate(unique_labels)}

  # Print unique labels
  print("Unique labels:", unique_labels)

  # Map labels to numeric values
  df['Label'] = df['Label'].map(label_mapping)

  # Write label mapping to a text file
  with open(os.path.join(folder_path, 'label_mappings.txt'), 'w') as f:
      for label, value in label_mapping.items():
          f.write(f"{label}: {value}\n")
  

  if show_plot:
      # Plot the distribution of classes
      plt.bar(df['Label'].unique(), df['Label'].value_counts(), edgecolor='black')
      plt.xticks(list(label_mapping.values()), labels=label_mapping.keys())
      plt.xlabel('Classes')
      plt.ylabel('Count') 
      plt.title('Distribution of Classes')
      plt.savefig(os.path.join(folder_path, "distribution_plot.png"))
      plt.close()

  X = df.drop('Label', axis=1)
  y = df['Label']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

  print("The train dataset size = ",X_train.shape)
  print("The test dataset size = ",X_test.shape)


  if existing_model is not None and os.path.exists(existing_model):
      model = continue_training(X_train, y_train, existing_model)
  else:
      model = train_model(X_train, y_train)
  
  joblib.dump(model, os.path.join(folder_path, 'random_forest_model.pkl'))


  X_train.to_csv(os.path.join(folder_path, 'X_train.csv'), index=False)
  X_test.to_csv(os.path.join(folder_path, 'X_test.csv'), index=False)
  y_train.to_csv(os.path.join(folder_path, 'y_train.csv'), index=False)
  y_test.to_csv(os.path.join(folder_path, 'y_test.csv'), index=False)

  print("Model saved successfully.")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train (create/update) RandomForestClassifier model')
    parser.add_argument('--input', '-I', help='Input CSV file')
    parser.add_argument('--output', '-O', help='Output directory')
    parser.add_argument('--plot', '-P', help='Show distribution of classes', default=False)
    parser.add_argument('--model', '-M', help='RF model to continue training')
    args = parser.parse_args()

    # Check if input and output files are provided
    if not args.input or not args.output:
        parser.error('Please provide both input and output files.')

    # Clean the CSV file
    run(args.input, args.output, args.plot, args.model)

if __name__ == '__main__':
    main()