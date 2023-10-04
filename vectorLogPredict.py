import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression

from collections import Counter
import numpy as np

# Step 1: Read and tokenize the data
labels = []
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
    #features.append((feature, current_pos))
    features.append(feature)
    labels.append(pos_tags[i])


# Step 4: Convert features to numerical vectors using DictVectorizer
dict_vectorizer = DictVectorizer()
X = dict_vectorizer.fit_transform(features)

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Step 6: Standardize the features (optional but can improve performance)
scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Train the logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Step 8: Make predictions on the test set
y_pred = logreg.predict(X_test)

# Step 9: Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

