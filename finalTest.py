import nltk
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from gensim.models import Word2Vec
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
            data.append((token, pos_tag))  # Ignore label in data

# Separate words and POS tags
words = [token for token, _ in data]
pos_tags = [pos_tag for _, pos_tag in data]

# Step 2: Train Word2Vec word embeddings
word_embedding_model = Word2Vec(sentences=[words], vector_size=100, window=5, min_count=1, sg=0)  # Adjust parameters as needed

# Step 3: Encode words using Word Embeddings
word_embeddings = np.array([word_embedding_model.wv[word] for word in words])

# Step 4: Encode POS tags using Label Encoding
word_encoder = LabelEncoder()
pos_tag_encoder = LabelEncoder()

word_encoded = word_encoder.fit_transform(words)
pos_tag_encoded = pos_tag_encoder.fit_transform(pos_tags)

# Step 5: Combine feature representations (including word embeddings)
# Combine word embeddings, word encodings, and POS tag encodings into a single feature vector

# Create a new list to store the previous word's POS tags
prev_pos_tags = ["<START>"] + pos_tags[:-1]

# Make sure <START> is present in the categories
if "<START>" not in pos_tag_encoder.classes_:
    pos_tag_encoder.classes_ = np.append(pos_tag_encoder.classes_, "<START>")

# Encode the previous POS tags just like the current POS tags
prev_pos_tag_encoded = pos_tag_encoder.transform(prev_pos_tags)

# Combine the previous POS tag representations with the current ones
combined_features = np.column_stack((word_embeddings, word_encoded, pos_tag_encoded, prev_pos_tag_encoded))

# Step 6: Generate dummy labels (since you don't have actual labels)
pos_tag_labels = pos_tag_encoder.transform(pos_tags)

# Step 7: Split the data into training and development sets
new_X_train, new_X_dev, new_y_train, new_y_dev = train_test_split(
    combined_features, pos_tag_labels, test_size=0.2, random_state=42
)

# Step 8: Classification using Naive Bayes
#bayesian_classifier = MultinomialNB()
#bayesian_classifier.fit(new_X_train, new_y_train)
#y_pred_bayesian = bayesian_classifier.predict(new_X_dev)

#print("Bayesian Classifier:")
#print("Accuracy:", accuracy_score(new_y_dev, y_pred_bayesian))
#print(classification_report(new_y_dev, y_pred_bayesian))

# Step 9: Classification using Logistic Regression
logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(new_X_train, new_y_train)
y_pred_logistic = logistic_regression.predict(new_X_dev)

print("Logistic Regression:")
print("Accuracy:", accuracy_score(new_y_dev, y_pred_logistic))
print(classification_report(new_y_dev, y_pred_logistic))

# Step 10: Classification using Support Vector Machines
svm_classifier = SVC()
svm_classifier.fit(new_X_train, new_y_train)
y_pred_svm = svm_classifier.predict(new_X_dev)

print("\nSupport Vector Machines:")
print("Accuracy:", accuracy_score(new_y_dev, y_pred_svm))
print(classification_report(new_y_dev, y_pred_svm))
