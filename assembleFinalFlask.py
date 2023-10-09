from collections import Counter
import re


log_tags = []
bayes_tags = [] 
svm_tags = []
words = []

def check_word(word):
    # Regular expression pattern to match words starting with '#', '@', or 'http'
    pattern1 = r'^(http)'
    pattern2 = r'#\S+'
    pattern3 = r'@[^ ]\S*'
    
    
    # Using re.match() to check if the word matches the pattern at the beginning
    if re.match(pattern1, word) or re.match(pattern2, word):
        return 1
    elif re.match(pattern3, word):
        return 2
    else:
        return 0

with open('Prediction_Logistic.txt', 'r') as file:
    words = [line.strip().split()[0] if len(line.strip().split()) >= 2 else '' for line in file]

# Read POS tags from logFLASK.txt, bayesFLASK.txt, and SVMFLASK.txt into lists
with open('Prediction_Logistic.txt', 'r') as file:
    log_tags = [line.strip().split()[1] if len(line.strip().split()) >= 2 else '' for line in file]

with open('Prediction_Bayes.txt', 'r') as file:
    bayes_tags = [line.strip().split()[1] if len(line.strip().split()) >= 2 else '' for line in file]

with open('Prediction_SVM.txt', 'r') as file:
    svm_tags = [line.strip().split()[1] if len(line.strip().split()) >= 2 else '' for line in file]

# Loop through tags and determine the most common tag appearing in at least two arrays, defaulting to SVM tag otherwise
final_tags = []
for log_tag, bayes_tag, svm_tag in zip(log_tags, bayes_tags, svm_tags):
    tags = [tag for tag in [log_tag, bayes_tag, svm_tag] if tag]
    tag_counts = Counter(tags)

    # Check if any tag appears in at least two arrays
    common_tags = [tag for tag, count in tag_counts.items() if count >= 2]

    if common_tags:
        most_common_tag = common_tags[0]  # Pick the first common tag
    else:
        most_common_tag = log_tag  # Default to Logistic Tag 

    final_tags.append(most_common_tag)

# Write the output to a new file
with open('FLASK.test.txt', 'w') as file:
    for word, tag in zip(words, final_tags):
        if(word == ''):
            file.write('\n')
        elif(check_word(word) == 1):
            file.write(f"{word} NN\n")
        elif(check_word(word) == 2):
            file.write(f"{word} NNP\n")
        else:
            file.write(f"{word} {tag}\n")
    file.write('\n')

print("Final tags written to FLASK.test.txt.")
