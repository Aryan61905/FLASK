import nltk

# Read words from the unlabeled test data
test_data = []
with open('unlabeled_test_test.txt', 'r') as file:
    for line in file:
        word = line.strip()  # Remove leading/trailing whitespace
        if word:
            test_data.append(word)

# Use NLTK's pre-trained POS tagger to get POS tags for the test data
pos_tags = nltk.pos_tag(test_data)

# Read original lines from the unlabeled test data again to preserve the structure
with open('unlabeled_test_test.txt', 'r') as file:
    lines = file.readlines()

# Write predictions to newFLASK.txt with tags added to each line
with open('newFLASK.txt', 'w') as file:
    line_index = 0
    for (_, tag) in pos_tags:
        # Skip empty lines in the original dataset
        while line_index < len(lines) and lines[line_index].strip() == '':
            file.write('\n')
            line_index += 1
        # Add the predicted tag to the current line
        file.write(f"{lines[line_index].strip()} {tag}\n")
        line_index += 1