import nltk
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from modelTraining import logistic_regression  # Import your trained model
import numpy as np

# Step 1: Read and tokenize the test data
test_data = []
with open('test.txt', 'r') as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:
            token, pos_tag, _ = line.split()  # Ignore label
            test_data.append((token, pos_tag))  # Ignore label in data

# Separate words and POS tags
test_words = [token for token, _ in test_data]
test_pos_tags = [pos_tag for _, pos_tag in test_data]

# Encode words and POS tags using Label Encoding (use the same encoders from training data)
test_word_encoded = word_encoder.transform(test_words)
test_pos_tag_encoded = pos_tag_encoder.transform(test_pos_tags)

# Create a new list to store the previous word's POS tags
prev_test_pos_tags = ["<START>"] + test_pos_tags[:-1]

# Make sure <START> is present in the categories
if "<START>" not in pos_tag_encoder.classes_:
    pos_tag_encoder.classes_ = np.append(pos_tag_encoder.classes_, "<START>")

# Encode the previous POS tags just like the current POS tags
prev_test_pos_tag_encoded = pos_tag_encoder.transform(prev_test_pos_tags)

# Combine the previous POS tag representations with the current ones for test data
combined_test_features = np.column_stack((test_word_encoded, test_pos_tag_encoded, prev_test_pos_tag_encoded))

# Step 2: Use the trained classifiers to predict on the test data
y_pred_bayesian_test = bayesian_classifier.predict(combined_test_features)
y_pred_logistic_test = logistic_regression.predict(combined_test_features)
y_pred_svm_test = svm_classifier.predict(combined_test_features)

# Step 3: Evaluate the classifiers on the test data
from sklearn.metrics import accuracy_score, classification_report

print("Evaluation on Test Data:")

print("\nBayesian Classifier:")
print("Accuracy:", accuracy_score(test_labels, y_pred_bayesian_test))
print(classification_report(test_labels, y_pred_bayesian_test))

print("\nLogistic Regression:")
print("Accuracy:", accuracy_score(test_labels, y_pred_logistic_test))
print(classification_report(test_labels, y_pred_logistic_test))

print("\nSupport Vector Machines:")
print("Accuracy:", accuracy_score(test_labels, y_pred_svm_test))
print(classification_report(test_labels, y_pred_svm_test))