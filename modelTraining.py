print("at least")
from sklearn.naive_bayes import MultinomialNB
print("at least")
from sklearn.linear_model import LogisticRegression
print("at leas3")
from sklearn.svm import SVC
print("at least")
from sklearn.preprocessing import StandardScaler
print("at least")
#from featureExtraction import X_train, y_train
print("at least")
#from diffFeatureExtraction import new_X_train, new_y_train
print("at least")
from logFeatureExtraction import log_X_train, log_y_train



#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)

print("got here")

# Train the Bayesian Classifier
bayesian_classifier = MultinomialNB()
print('2')
#bayesian_classifier.fit(X_train, y_train)
#bayesian_classifier.fit(new_X_train, new_y_train)

# Train the logistic regression model
logistic_regression = LogisticRegression(max_iter=1000)
print('3')
#logistic_regression.fit(X_train, y_train)
logistic_regression.fit(log_X_train, log_y_train)
print("4")


# Train the Support Vector Machines (SVM) model
svm_classifier = SVC()
print("5")

#svm_classifier.fit(X_train, y_train)
#svm_classifier.fit(new_X_train, new_y_train)


