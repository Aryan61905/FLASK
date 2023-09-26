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
from gensim.models import Word2Vec
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

# Step 2: Train Word2Vec word embeddings
word_embedding_dim = 100  # You can adjust the dimensionality
word2vec_model = Word2Vec(sentences=[words], vector_size=word_embedding_dim, window=5, min_count=1)

# Step 3: Encode labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Step 4: Create feature vectors using Word2Vec embeddings
word_embeddings = [word2vec_model.wv[word] if word in word2vec_model.wv else np.zeros(word_embedding_dim) for word in words]
word_embeddings = np.array(word_embeddings)

# Step 5: Combine feature representations
# Combine word embeddings with POS tags
pos_tag_encoder = LabelEncoder()
pos_tags_encoded = pos_tag_encoder.fit_transform(pos_tags)

# Combine word embeddings and POS tag encodings
combined_features = np.hstack((word_embeddings, pos_tags_encoded.reshape(-1, 1)))
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