import argparse
from typing import Optional
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def predict(path_to_model: str, path_to_features: str, path_to_labels: Optional[str]):
  data=pd.read_csv(path_to_features)
  
  model = joblib.load(path_to_model)
  predictions = model.predict(data)

  # Print the prediction counts for each label
  prediction_counts = pd.Series(predictions).value_counts()
  for label, count in prediction_counts.items():
    print(f"Number of rows predicted as '{label}': {count}")

  if path_to_labels is not None:
    labels=pd.read_csv(path_to_labels)
    series = labels['Label'].value_counts()
    for label, count in series.items():
        print(f"Number of rows labeled as '{label}': {count}")

    print('\nPrediction Metrics:')
    print(f'Accuracy: {round(accuracy_score(labels, predictions), 4)}')
    print(f'F1 Score: {round(f1_score(labels, predictions, average="weighted", zero_division=1), 4)}')
    print(f'Precision: {round(precision_score(labels, predictions, average="weighted", zero_division=1), 4)}')
    print(f'Recall: {round(recall_score(labels, predictions, average="weighted"), 4)}')


def main():
  # Parse command-line arguments
  parser = argparse.ArgumentParser(description='Predict labels for features')
  parser.add_argument('--model', '-M', help='Model')
  parser.add_argument('--features', '-F', help='Features file')
  parser.add_argument('--labels', '-L', help='Optional: Labels file to show statistics')
  args = parser.parse_args()

  # Check if input and output files are provided
  if not args.features and not args.model:
      parser.error('Please provide path to file containing features and the model.')

  # Clean the CSV file
  predict(args.model, args.features, args.labels)

if __name__ == '__main__':
    main()