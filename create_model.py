import joblib
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def create_model(input_file: str, output_dir: str, show_plot: bool):
  folder_path = os.path.join(os.getcwd(), output_dir)

  if not os.path.exists(folder_path):
      os.makedirs(folder_path)


  df=pd.read_csv(input_file)
  df['Label'] = df['Label'].map({'BENIGN': 0, 'DDoS': 1, 'PortScan': 2, 'Infiltration': 3 }) # TODO: Web attack


  if show_plot:
    plt.bar([0, 1], df['Label'].value_counts(), edgecolor='black')
    plt.xticks([0, 1], labels=['BENIGN=0', 'DDoS=1'])
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

  rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
  rf_model.fit(X_train, y_train)

  
  joblib.dump(rf_model, os.path.join(folder_path, 'random_forest_model.pkl'))
  X_train.to_csv(os.path.join(folder_path, 'X_train.csv'), index=False)
  X_test.to_csv(os.path.join(folder_path, 'X_test.csv'), index=False)
  y_train.to_csv(os.path.join(folder_path, 'y_train.csv'), index=False)
  y_test.to_csv(os.path.join(folder_path, 'y_test.csv'), index=False)

  print("Model saved successfully.")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Create and train RandomForestClassifier model')
    parser.add_argument('--input', '-I', help='Input CSV file')
    parser.add_argument('--output', '-O', help='Output directory')
    parser.add_argument('--plot', '-P', help='Show distribution of classes', default=False)
    args = parser.parse_args()

    # Check if input and output files are provided
    if not args.input or not args.output:
        parser.error('Please provide both input and output files.')

    # Clean the CSV file
    create_model(args.input, args.output, args.plot)

if __name__ == '__main__':
    main()