import nltk
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# Step 1: Read and tokenize the data
data = []
with open('train.txt', 'r') as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:
            token, pos_tag, label = line.split()  # Include label
            data.append((token, pos_tag, label))  # Include label in data

# Separate words, POS tags, and labels
words = [token for token, _, _ in data]
pos_tags = [pos_tag for _, pos_tag, _ in data]
labels = [label for _, _, label in data]

# Step 2: Encode words and POS tags
# You can use one-hot encoding for both words and POS tags
word_encoder = OneHotEncoder(sparse_output=True, sparse=False)
pos_tag_encoder = OneHotEncoder(sparse_output=True, sparse=False)

word_encoded = word_encoder.fit_transform(np.array(words).reshape(-1, 1))
pos_tag_encoded = pos_tag_encoder.fit_transform(np.array(pos_tags).reshape(-1, 1))

# Step 3: Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Step 4: Combine feature representations
# Combine word and POS tag representations into a single feature vector

# Create a new list to store the previous word's POS tags
prev_pos_tags = ["<START>"] + pos_tags[:-1]

# Make sure <START> is present in the categories
if "<START>" not in pos_tag_encoder.categories_[0]:
    pos_tag_encoder.categories_[0] = np.append(pos_tag_encoder.categories_[0], "<START>")

# Encode the previous POS tags just like the current POS tags
prev_pos_tag_encoded = pos_tag_encoder.transform(np.array(prev_pos_tags).reshape(-1, 1))

# Combine the previous POS tag representations with the current ones
combined_features = np.hstack((word_encoded, pos_tag_encoded, prev_pos_tag_encoded))

# Step 5: Use combined_features as input for your classifier
# Now, combined_features contains the numerical vectors for each token,
# and you can use it as input for your classifier.

# Example usage:
log_X_train, log_X_dev, log_y_train, log_y_dev = train_test_split(combined_features, labels_encoded, test_size=0.2, random_state=42)
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_dev)