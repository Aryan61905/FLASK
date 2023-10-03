import nltk

# Read words from the unlabeled test data
test_data = []
with open('unlabeled_test_test.txt', 'r') as file:
    for line in file:
        word = line.strip()  # Remove leading/trailing whitespace
        if word:
            test_data.append(word)

# Use NLTK's pre-trained POS tagger
pos_tags = nltk.pos_tag(test_data)

# Write predictions to FLASK.txt
with open('newFLASK.txt', 'w') as file:
    for word, tag in pos_tags:
        file.write(f"{word} {tag}\n")