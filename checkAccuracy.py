import nltk
from sklearn.metrics import  accuracy_score

our_tags = []
with open('logFlask.txt', 'r') as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:
            _, tag = line.split()  # Extract POS tag from train data
            our_tags.append(tag)


# Read words from the unlabeled test data
test_data = []
with open('unlabeled_test_test.txt', 'r') as file:
    for line in file:
        word = line.strip()  # Remove leading/trailing whitespace
        if word:
            test_data.append(word)

# Use NLTK's pre-trained POS tagger
pos_tags = nltk.pos_tag(test_data)
new_pos_tags = []
for word, tag in pos_tags:
    new_pos_tags.append(tag)


# Calculate and print accuracy for NLTK POS tags
nltk_accuracy = accuracy_score(our_tags, new_pos_tags)
print(f"NLTK Accuracy: {nltk_accuracy:.2f}")