from sklearn.preprocessing import StandardScaler
from evaluation import *
#implement a standard scalar on gradient descent for optimal performance
sc=StandardScaler()
sc.fit(X_train) # estimatd parameter eta and standard deviation for each training data
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)
