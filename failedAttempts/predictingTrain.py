import nltk
import os
os.system('pip3 install spacy')
spacy.load('en_core_web_sm')
from sklearn.metrics import confusion_matrix, accuracy_score

# Read words and old POS tags from the labeled train data
train_data = []
old_pos_tags = []
with open('train.txt', 'r') as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:
            token, pos_tag, _ = line.split()  # Extract POS tag from train data
            train_data.append(token)
            old_pos_tags.append(pos_tag)

# NLTK POS tagging
nltk_tags = nltk.pos_tag(train_data)
nltk_pos_tags = [tag for _, tag in nltk_tags]

# SpaCy POS tagging
nlp = spacy.load('en_core_web_sm')
spacy_doc = nlp(' '.join(train_data))
spacy_pos_tags = [token.pos_ for token in spacy_doc]

# Write NLTK predictions to test1.txt
with open('test1.txt', 'w') as file:
    for word, tag in nltk_tags:
        file.write(f"{word} {tag}\n")

# Write SpaCy predictions to test2.txt
with open('test2.txt', 'w') as file:
    for word, tag in zip(train_data, spacy_pos_tags):
        file.write(f"{word} {tag}\n")

# Calculate and print accuracy for NLTK POS tags
nltk_accuracy = accuracy_score(old_pos_tags, nltk_pos_tags)
print(f"NLTK Accuracy: {nltk_accuracy:.2f}")

# Calculate and print accuracy for SpaCy POS tags
spacy_accuracy = accuracy_score(old_pos_tags, spacy_pos_tags)
print(f"SpaCy Accuracy: {spacy_accuracy:.2f}")

# Print confusion matrix for NLTK POS tags
nltk_cm = confusion_matrix(old_pos_tags, nltk_pos_tags)
print("NLTK Confusion Matrix:")
print(nltk_cm)

# Print confusion matrix for SpaCy POS tags
spacy_cm = confusion_matrix(old_pos_tags, spacy_pos_tags)
print("SpaCy Confusion Matrix:")
print(spacy_cm)
