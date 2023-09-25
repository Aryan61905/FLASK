from sklearn.metrics import accuracy_score, classification_report
print("here")
#from featureExtraction import X_dev, y_dev
#from modelTraining import bayesian_classifier  # Import your trained model

from modelTraining import logistic_regression  # Import your trained model
print("1")
#from diffFeatureExtraction import new_X_dev, new_y_dev
from logFeatureExtraction import log_X_dev, log_y_dev
print("2")

# Predict POS tags for the development set using the trained model
#y_pred_bayesian = bayesian_classifier.predict(X_dev)
#y_pred_bayesian = bayesian_classifier.predict(new_X_dev)

# Predict POS tags for the development set using logistic regression
#y_pred_logistic = logistic_regression.predict(X_dev_transformed)
y_pred_logistic = logistic_regression.predict(log_X_dev)
print("3")

# Predict POS tags for the development set using SVM
#y_pred_svm = svm_classifier.predict(X_dev_transformed)

# Evaluate the performance of the model
#print("Bayesian Classifier:")
#print("Accuracy:", accuracy_score(y_dev, y_pred_bayesian))
#print(classification_report(y_dev, y_pred_bayesian))

#print("Bayesian Classifier:")
#print("Accuracy:", accuracy_score(new_y_dev, y_pred_bayesian))
#print(classification_report(new_y_dev, y_pred_bayesian))

# Evaluate the performance of logistic regression
#print("Logistic Regression:")
#print("Accuracy:", accuracy_score(y_dev, y_pred_logistic))
#print(classification_report(y_dev, y_pred_logistic))

print("Logistic Regression:")
print("Accuracy:", accuracy_score(log_y_dev, y_pred_logistic))
print(classification_report(log_y_dev, y_pred_logistic))

# Evaluate the performance of SVM
#print("\nSupport Vector Machines:")
#print("Accuracy:", accuracy_score(y_dev, y_pred_svm))
#print(classification_report(y_dev, y_pred_svm))