from sklearn.linear_model import Perceptron
from standard_scaler_class import *
ppn=Perceptron()
ppn.fit(X_train_std,y_train)
y_pred=ppn.predict(X_test_std)
print('Misclassified examples: %d' %(y_test != y_pred).sum())