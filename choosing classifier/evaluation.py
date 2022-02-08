from cProfile import label
from sklearn.model_selection import train_test_split
import numpy as np
from using_sktlearn import *
X_train,X_test,y_train,y_test=train_test_split( X,y,test_size=0.3,random_state=1,stratify=y)#by doing this we split the test data o 30% and training data to 70%
print('labels counts in y:',np.bincount(y))
print('labels in y_train:',np.bincount(y_train))
print('labels in y_test:',np.bincount(y_test))
