import nltk
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
print("1")

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
print("2")
# Step 2: Encode words and POS tags
# You can use one-hot encoding for both words and POS tags
word_encoder = OneHotEncoder(sparse_output=True, sparse=False)
pos_tag_encoder = OneHotEncoder(sparse_output=True, sparse=False)

word_encoded = word_encoder.fit_transform(np.array(words).reshape(-1, 1))
pos_tag_encoded = pos_tag_encoder.fit_transform(np.array(pos_tags).reshape(-1, 1))

# Step 3: Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
print("3")
# Step 4: Combine feature representations
# Combine word and POS tag representations into a single feature vector

combined_features = np.hstack((word_encoded, pos_tag_encoded))

# Step 5: Use combined_features as input for your classifier
# Now, combined_features contains the numerical vectors for each token,
# and you can use it as input for your classifier.
print("4")  
# Example usage:
X_train, X_dev, y_train, y_dev = train_test_split(combined_features, labels_encoded, test_size=0.2, random_state=42)
svm_classifier = SVC()
print("5")  
svm_classifier.fit(X_train, y_train)
print("6")
y_pred_svm = svm_classifier.predict(X_dev)
print("7")
print("\nSupport Vector Machines:")
print("Accuracy:", accuracy_score(y_dev, y_pred_svm))
print(classification_report(y_dev, y_pred_svm))
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_dev)