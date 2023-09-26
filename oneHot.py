import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Step 1: Read and tokenize the data
data = []
with open('train.txt', 'r') as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:
            token, pos_tag, _ = line.split()  # Ignore label
            data.append((token, pos_tag))  # Ignore label in data

# Separate words and POS tags
words = [token for token, _ in data]
pos_tags = [pos_tag for _, pos_tag in data]

# Step 2: Encode words and POS tags using One-Hot Encoding
word_encoder = OneHotEncoder(sparse=False)
pos_tag_encoder = OneHotEncoder(sparse=False)

word_encoded = word_encoder.fit_transform(np.array(words).reshape(-1, 1))
pos_tag_encoded = pos_tag_encoder.fit_transform(np.array(pos_tags).reshape(-1, 1))

# Step 3: Combine feature representations
# Combine word and POS tag representations into a single feature vector

# Create a new list to store the previous word's POS tags
prev_pos_tags = ["<START>"] + pos_tags[:-1]

# Make sure <START> is present in the categories
if not any("<START>" in category for category in pos_tag_encoder.categories_):
    pos_tag_encoder.categories_ = [np.append(category, "<START>") for category in pos_tag_encoder.categories_]

# Encode the previous POS tags just like the current POS tags
prev_pos_tag_encoded = pos_tag_encoder.transform(np.array(prev_pos_tags).reshape(-1, 1))

# Combine the previous POS tag representations with the current ones
combined_features = np.concatenate((word_encoded, pos_tag_encoded, prev_pos_tag_encoded), axis=1)

# Step 4: Generate dummy labels
# Since you don't have actual labels, you can generate dummy labels for training and development data.
# Here, I'm generating random integers as dummy labels.
np.random.seed(42)

dummy_labels = np.random.randint(0, 2, size=len(combined_features))  # Generate random 0s and 1s as labels

# Step 5: Split the data into training and development sets
new_X_train, new_X_dev, new_y_train, new_y_dev = train_test_split(
    combined_features, dummy_labels, test_size=0.2, random_state=42
)

from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
bayesian_classifier = MultinomialNB()
from sklearn.linear_model import LogisticRegression
print("at leas3")
from sklearn.svm import SVC
print('2')
bayesian_classifier.fit(new_X_train,new_y_train)
y_pred_bayesian = bayesian_classifier.predict(new_X_dev)
print("Bayesian Classifier:")
print("Accuracy:", accuracy_score(new_y_dev, y_pred_bayesian))
print(classification_report(new_y_dev, y_pred_bayesian))

logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(new_X_train, new_y_train)
y_pred_logistic = logistic_regression.predict(new_X_dev)
print("Logistic Regression:")
print("Accuracy:", accuracy_score(new_y_dev, y_pred_logistic))
print(classification_report(new_y_dev, y_pred_logistic))

svm_classifier = SVC()
print("5")
svm_classifier.fit(new_X_train, new_y_train)
y_pred_svm = svm_classifier.predict(new_X_dev)
print("\nSupport Vector Machines:")
print("Accuracy:", accuracy_score(new_y_dev, y_pred_svm))
print(classification_report(new_y_dev, y_pred_svm))
