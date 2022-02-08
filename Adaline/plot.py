import numpy as np
import matplotlib.pyplot as plt
from implementation import *
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,4))
ada1=adaline(n_inter=10,eta=0.01).fit(X,y)
ax[0].plot(range(1,len(ada1.cost_)+1),np.log10(ada1.cost_),marker='0')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared error)')
ax[0].set_title('Adaline-Learning rate 0.01')
ada2=adaline(n_inter=10,eta=0.0001).fit(X,y)
ax[1].plot(range(1,len(ada2.cost_)+1),ada2.cost_,marker='0')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared error')
ax[1].set_title('Adaline-learning rate 0.0001')
plt.show()