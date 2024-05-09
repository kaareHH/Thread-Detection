# detection
This project contains tools to:
- merge_datasets.py: Merge any number of datasets. Simply place the desired sets to merge within the same folder
- preprocess.py: Remove infinite/inf/null values from the datasets
- create_model.py: Train a RandomForestClassifier based on a csv file with features, and store the result in a folder
- predict.py: Use a model to predict labels from a csv file with features

## Dataset
Datasets: https://www.unb.ca/cic/datasets/ids-2017.html
Helpful Article: https://www.labellerr.com/blog/ddos-attack-detection/amp/

## Good to know
https://stackoverflow.com/questions/74613826/how-do-i-apply-my-random-forest-classifier-to-an-unlabelled-dataset


## Hello World example
1. Download the datasets from the link above
2. Transfer the datasets to the root of this project

3. Preprocess the data
```
python .\preprocess.py -I .\MachineLearningCSV\MachineLearningCVE\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv -O pcap.csv      
```

4. Create and train the model (-P true prints a distribution class plot for the labels to the destination folder)
```
python .\create_model.py -I .\pcap.csv -O output -P true
```

5. Use the model to predict on the test data created during training (-L tells the prediction script to include prediction metrics).
```
python .\predict.py -M .\output\random_forest_model.pkl -F .\output\X_test.csv -L .\output\y_test.csv 
```


# RandomForestClassifier: A Layman's Explanation - by ChatGpt

### What is RandomForestClassifier?
`RandomForestClassifier` is a machine learning algorithm used for classification tasks. It's part of a broader class of algorithms known as ensemble methods. Ensemble methods work by combining the predictions of multiple individual models to improve overall performance.

### How does RandomForestClassifier work?
Imagine you're trying to make a big decision, like where to go on vacation. Instead of relying on just one friend's advice, you ask multiple friends who have different perspectives and experiences. You then aggregate their opinions to make a more informed decision. RandomForestClassifier works similarly.

Here's a simplified explanation of how it works:

1. `Build Many Decision Trees`: RandomForestClassifier builds a "forest" of decision trees during training. Each decision tree is like asking one friend for advice.
2. `Randomness`: It introduces randomness in two ways:
   - `Random Sampling`: Each decision tree is trained on a random subset of the data (called bootstrap samples). This ensures that each tree sees a slightly different perspective of the data.
   - `Random Subset of Features`: At each split in a decision tree, only a random subset of features is considered. This helps to create diversity among the trees.
3. `Voting`: When making predictions, each decision tree in the forest gets a vote. The final prediction is determined by a majority vote (for classification) or averaging (for regression) of the individual tree predictions.


### Parameters of RandomForestClassifier:
1. `n_estimators`: Number of decision trees in the forest. Increasing this parameter can improve performance, but also increases computational cost.
2. `max_depth`: Maximum depth of each decision tree. Controls the maximum depth of the tree. Increasing it may lead to overfitting.
3. `min_samples_split`: Minimum number of samples required to split an internal node. Increasing this parameter prevents the tree from splitting too early, which can help prevent overfitting.
4. `min_samples_leaf`: Minimum number of samples required to be at a leaf node. Increasing it prevents the tree from creating nodes with very few samples, which can help prevent overfitting.
5. `max_features`: Number of features to consider when looking for the best split. It determines the randomness in feature selection. Lower values introduce more randomness.
6. `random_state`: Controls the randomness during model training. Setting it to a fixed value ensures reproducibility.

### What changing these parameters would do:
- `Increasing n_estimators`: Usually improves performance but increases computational cost.
- Increasing max_depth: Can lead to more complex decision boundaries and potentially overfitting.
- `Increasing min_samples_split and min_samples_leaf`: Makes the model more conservative, preventing overfitting.
- `Decreasing max_features`: Increases randomness and reduces overfitting but may decrease predictive power if set too low.

In essence, RandomForestClassifier is like a group of friends making a decision together, where each friend has a slightly different perspective, and they vote to make a final decision. The parameters control how many friends are involved, how deeply they think, and how much they listen to each other. Adjusting these parameters can help balance between complexity, accuracy, and computational cost.