import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
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

# Step 3: Extract features using NLTK
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

    # Combine features as a dictionary
    feature = {
        'word': current_word,
        'prev_word': previous_word,
        'next_word': next_word,
        'is_capitalized': is_capitalized,
        'prefix1': prefix1,
        'prefix2': prefix2,
        'prefix3': prefix3,
        'suffix1': suffix1,
        'suffix2': suffix2,
        'suffix3': suffix3,
        'word_length': word_length,
        'has_digits': has_digits,
        'has_punctuation': has_punctuation,
    }
    features.append((feature, current_pos))

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, pos_tags, test_size=0.2, random_state=42)

# Step 5: Define NLTK classifiers and train them
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression

# Logistic Regression Classifier
print("-1")
logistic_classifier = SklearnClassifier(LogisticRegression(solver="liblinear"))
logistic_classifier.train(X_train)
print("Logistic Regression Classifier Accuracy:", nltk.classify.accuracy(logistic_classifier, X_test))


# Read words from the unlabeled test data
test_data = []
with open('unlabeled_test_test.txt', 'r') as file:
    for line in file:
        word = line.strip()  # Remove leading/trailing whitespace
        if word:
            test_data.append(word)

# Extract features from the test data
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

    # Combine features as a dictionary
    feature = {
        'word': current_word,
        'prev_word': previous_word,
        'next_word': next_word,
        'is_capitalized': is_capitalized,
        'prefix1': prefix1,
        'prefix2': prefix2,
        'prefix3': prefix3,
        'suffix1': suffix1,
        'suffix2': suffix2,
        'suffix3': suffix3,
        'word_length': word_length,
        'has_digits': has_digits,
        'has_punctuation': has_punctuation,
    }
    test_features.append(feature)

# Use the trained logistic regression classifier to make predictions on the test data
y_pred_logistic = [logistic_classifier.classify(feature) for feature in test_features]

with open('unlabeled_test_test.txt', 'r') as file:
    lines = file.readlines()

# Write predictions to logFLASK.txt with tags added to each line
with open('logFLASK.txt', 'w') as file:
    line_index = 0
    for word, tag in zip(test_data, y_pred_logistic):
        # Skip empty lines in the original dataset
        while line_index < len(lines) and lines[line_index].strip() == '':
            file.write('\n')
            line_index += 1
        # Add the predicted tag to the current line
        file.write(f"{lines[line_index].strip()} {tag}\n")
        line_index += 1