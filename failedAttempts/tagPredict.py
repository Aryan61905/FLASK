import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from collections import Counter
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

# Step 2: Create a list of previous POS tags
prev_pos_tags = ["<START>"] + pos_tags[:-1]

# Step 3: Extract capitalization, prefixes, and suffixes as features
features = []
for i in range(len(words)):
    current_word = words[i]
    current_pos = pos_tags[i]
    previous_pos = prev_pos_tags[i]
    previous_word = words[i - 1] if i > 0 else "<START_WORD>"  # Include "<START_WORD>" for the first word
    prev_prev_word = words[i - 2] if i >= 2 else "<START_WORD>"  # Include "<START_WORD>" for the first two words
    next_word = words[i + 1] if i < len(words) - 1 else "<END_WORD>"  # Include "<END_WORD>" for the last word

    # Capitalization feature
    is_capitalized = int(current_word[0].isupper())

    # Prefixes (e.g., first 3 characters)
    prefix1 = current_word[:1]
    prefix2 = current_word[:2]
    prefix3 = current_word[:3]

    # Suffixes (e.g., last 3 characters)
    suffix1 = current_word[-1:]
    suffix2 = current_word[-2:]
    suffix3 = current_word[-3:]
    # Word length
    word_length = len(current_word)

    # Presence of digits
    has_digits = int(any(char.isdigit() for char in current_word))

    # Presence of punctuation
    has_punctuation = int(any(char in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" for char in current_word))

    # Word frequency
    word_frequency = Counter[current_word]

    # POS tags of previous two words
    prev_prev_pos = prev_pos_tags[i - 2] if i >= 2 else "<START>"
    prev_pos = prev_pos_tags[i - 1] if i >= 1 else "<START>"

    # Combine features
    feature = f"{current_word} {previous_word} {next_word} {is_capitalized} {prefix1} {prefix2} {prefix3} {suffix1} {suffix2} {suffix3} {word_length} {has_digits} {has_punctuation}"
    features.append(feature)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, pos_tags, test_size=0.2, random_state=42)

# Step 5: Vectorize the features using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Step 6: Train models to predict POS tags
# Multinomial Naive Bayes
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vectorized, y_train)
y_pred_nb = nb_classifier.predict(X_test_vectorized)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Multinomial Naive Bayes Accuracy:", accuracy_nb)

# Logistic Regression
logistic_classifier = LogisticRegression(max_iter=1000)
logistic_classifier.fit(X_train_vectorized, y_train)
y_pred_logistic = logistic_classifier.predict(X_test_vectorized)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print("Logistic Regression Accuracy:", accuracy_logistic)

# Support Vector Machine (SVM)
# Add feature scaling with StandardScaler
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train_vectorized.toarray())  # Convert to dense array for scaling
#X_test_scaled = scaler.transform(X_test_vectorized.toarray())

#svm_classifier = SVC(max_iter=200)
#svm_classifier.fit(X_train_scaled, y_train)
#y_pred_svm = svm_classifier.predict(X_test_scaled)
#accuracy_svm = accuracy_score(y_test, y_pred_svm)
#print("SVM Accuracy:", accuracy_svm)
#svm_classifier = SVC()
#svm_classifier.fit(X_train_scaled, y_train)
#y_pred_svm = svm_classifier.predict(X_test_scaled)
#accuracy_svm = accuracy_score(y_test, y_pred_svm)
#print("SVM Accuracy:", accuracy_svm)


print("here")
# Step 1: Read and tokenize the test data
test_data = []
test_pos_tags = []  # List to store the actual POS tags
with open('test.txt', 'r') as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:
            token, pos_tag, _ = line.split()  # Extract POS tag from test data
            test_data.append(token)
            test_pos_tags.append(pos_tag)


# Step 2: Extract features from the test data
print("1")
test_features = []
for i in range(len(test_data)):
    current_word = test_data[i]
    previous_word = test_data[i - 1] if i > 0 else "<START_WORD>"  # Include "<START_WORD>" for the first word
    prev_prev_word = test_data[i - 2] if i >= 2 else "<START_WORD>"  # Include "<START_WORD>" for the first two words
    next_word = test_data[i + 1] if i < len(test_data) - 1 else "<END_WORD>"  # Include "<END_WORD>" for the last word

    # Capitalization feature
    is_capitalized = int(current_word[0].isupper())

    # Prefixes (e.g., first 3 characters)
    prefix1 = current_word[:1]
    prefix2 = current_word[:2]
    prefix3 = current_word[:3]

    # Suffixes (e.g., last 3 characters)
    suffix1 = current_word[-1:]
    suffix2 = current_word[-2:]
    suffix3 = current_word[-3:]

    # Word length
    word_length = len(current_word)

    # Presence of digits
    has_digits = int(any(char.isdigit() for char in current_word))

    # Presence of punctuation
    has_punctuation = int(any(char in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" for char in current_word))

    # Combine features
    feature = f"{current_word} {previous_word} {next_word} {is_capitalized} {prefix1} {prefix2} {prefix3} {suffix1} {suffix2} {suffix3} {word_length} {has_digits} {has_punctuation}"
    test_features.append(feature)

    # Step 3: Vectorize the test features using the same CountVectorizer
X_test_vectorized = vectorizer.transform(test_features)

# Step 4: Use your trained models to make predictions on the test data
print("3")
y_pred_nb = nb_classifier.predict(X_test_vectorized)
y_pred_logistic = logistic_classifier.predict(X_test_vectorized)

# Standardize test data for SVM
#X_test_scaled = scaler.transform(X_test_vectorized.toarray())  # Convert to dense array for scaling
#y_pred_svm = svm_classifier.predict(X_test_scaled)

# Step 5: Evaluate the performance of your models
# You can calculate accuracy by comparing the predicted POS tags with the actual POS tags in test_pos_tags.

accuracy_nb = accuracy_score(test_pos_tags, y_pred_nb)
accuracy_logistic = accuracy_score(test_pos_tags, y_pred_logistic)
#accuracy_svm = accuracy_score(test_pos_tags, y_pred_svm)

print("Multinomial Naive Bayes Accuracy:", accuracy_nb)
print("Logistic Regression Accuracy:", accuracy_logistic)
#print("SVM Accuracy:", accuracy_svm)