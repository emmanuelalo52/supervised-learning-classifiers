#calculating the classification accuracy with classification misclassification
from sklearn.metrics import accuracy_score
from standard_scaler_class import *
from training_perceptron import *
print('Accuracy: %.3f' % accuracy_score(y_test,y_pred))
print('Accuracy: %.3f' % ppn.score(X_test_std,y_test))