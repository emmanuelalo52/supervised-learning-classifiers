import numpy as np
from numpy.lib.index_tricks import nd_grid
class adaline(object):
    """adaline parameters includes learning rate(0-1), n_inter(int passes over the training set),
        random weight initialization. Attributes: w_: 1d array weights after fitting. cost_: ist SSE value in each epoch"""
    def __init__(self,eta=0.01,n_inter=50,random_sate=1):
        self.eta=eta
        self.n_inter=n_inter
        self.random_state=random_sate
    def fit(self,X,y):
        """fit trainig data parameters : X:array like, shape=[]n_examples,n_examples,n_features, Training vectors, where n_examples is the number 
            of examples and n_features is the number of features. y : array like, shape=[n_examples] Target values. Returns- self: object"""
        rgen=np.random.RandomState(self.random_state)
        self.w_=rgen.normal(loc=0.0,scale=0.01,size=1 + X.shape[1])
        self.cost_=[]
        for i in range(self.n_inter):
            net_input=self.net_input(X)
            output=self.activation(net_input)
            errors=(y-output)
            self.w_[1:]+=self.eta*X.T.dot(errors)
            self.w_[0]+=self.eta*errors.sum()
            cost=(errors**2).sum()/2.0
            self.cost_.append(cost)
            return self
    def  net_input(self,X):
        """calculating our net input"""
        return np.dot(X,self.w_[1:]) + self.w_[0]
    def activation(Self,X):
        #linear activation
        return X
    def predict(self,X):
        #return class label after every unit step
        return np.where(self.activation(self.net_input(X))>=0.0,1,-1)