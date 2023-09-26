import nltk
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Step 1: Read and tokenize the data
data = []
with open('train.txt', 'r') as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:
            token, pos_tag, _ = line.split()  # Ignore label
            data.append(token)  # Only include tokens in data

print("Step 1: Data Loaded and Tokenized")

# Step 2: Extract features (next part of speech, prefixes/suffixes, and capitalization)
print("1")
prefixes = [token[:3] for token in data]  # Extract the first 3 characters as a prefix feature
print("2")
suffixes = [token[-3:] for token in data]  # Extract the last 3 characters as a suffix feature
print("3")
capitalization = [int(token[0].isupper()) for token in data]  # 1 if capitalized, 0 otherwise

print("Step 2: Features Extracted")

# Step 3: Encode words and features using TF-IDF and other features
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Adjust the number as needed
tfidf_encoded = tfidf_vectorizer.fit_transform(data)

# Combine TF-IDF features with the extracted features
tfidf_encoded = np.hstack([tfidf_encoded.toarray(),
                           np.array(prefixes).reshape(-1, 1), np.array(suffixes).reshape(-1, 1),
                           np.array(capitalization).reshape(-1, 1)])

print("Step 3: Features Encoded")

# Step 4: Generate dummy labels (random)
np.random.seed(42)
dummy_labels = np.random.randint(0, 2, size=tfidf_encoded.shape[0])  # Generate random 0s and 1s as labels

print("Step 4: Dummy Labels Generated")

# Step 5: Split the data into training and development sets
X_train, X_dev, y_train, y_dev = train_test_split(
    tfidf_encoded, dummy_labels, test_size=0.2, random_state=42
)

print("Step 5: Data Split into Training and Development Sets")

# Step 6: Train and evaluate classifiers
bayesian_classifier = MultinomialNB()
bayesian_classifier.fit(X_train, y_train)
y_pred_bayesian = bayesian_classifier.predict(X_dev)
print("Bayesian Classifier:")
print("Accuracy:", accuracy_score(y_dev, y_pred_bayesian))
print(classification_report(y_dev, y_pred_bayesian))

print("Bayesian Classifier Evaluation Complete")

logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(X_train, y_train)
y_pred_logistic = logistic_regression.predict(X_dev)
print("\nLogistic Regression:")
print("Accuracy:", accuracy_score(y_dev, y_pred_logistic))
print(classification_report(y_dev, y_pred_logistic))

print("Logistic Regression Evaluation Complete")

svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_dev)
print("\nSupport Vector Machines:")
print("Accuracy:", accuracy_score(y_dev, y_pred_svm))
print(classification_report(y_dev, y_pred_svm))

print("Support Vector Machines Evaluation Complete")
