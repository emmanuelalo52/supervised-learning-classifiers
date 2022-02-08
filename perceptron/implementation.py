import numpy as np
class perceptron(object):
    """Perceptron classifier
    what we need are : float,learning rate,n_inter:int passes over the trauning set ,random numbers generator seeed for random weight initializtion
    attributes ;w:id_array Weights after fitting errors_:list number of misclassification (updates) each epochs"""
    def __init__(self,eta=0.01,n_inter=50,random_state=1):
        self.eta=eta
        self.n_inter=n_inter
        self.random_state=random_state
    def fit(self,X,y):
        """fit training data set
        parameters : X :{array-like},shape=[]n_examples,n_features",Training vectors,where n_examples is the number of examples and n_features
        is the number of features. y: array-like , shape =[n_examples]Target values.Returns self:object"""
        rgen=np.random.RandomState(self.random_state)
        self.w_=rgen.normal(loc=0.0,scale=0.01,size=1+X.shape[1])
        self.errors_=[]
        for _ in range(self.n_inter):
            errors=0
        for xi,target in zip(X,y):
            update=self.eta*(target-self.predict(xi))
            self.w_[1:]+=update*xi
            self.w_[0]+=update
            errors+=int(update!=0.0)
            self.errors_.append(errors)
            return self
    def net_input(self,X):
        #calculate z or net input 
        return np.dot(X,self.w_[1:])+self.w_[0]
    def predict(self,X):
        #return class label after unit step
        return np.where(self.net_input(X)>=0.0,1,-1)