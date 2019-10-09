#This class is used to train the preceptron
import numpy as np 
import random 

class Perceptron():

    def __init__(self,x_train,y_train,eta,epoch):
        self.x_train = x_train # Training data
        self.y_train = y_train #Training labels
        self.eta = eta #Learning rate
        self.epoch = epoch #Maximum number of trainig epochs

    #Function to train and that returns the weights
    def train(self):

        w = np.zeros((1,(self.x_train.ndim)+1))
        for i  in range((self.x_train.ndim)+1):
            w[:,i] = (random.random())
        
        self.x_train = np.insert(self.x_train, len(self.x_train[0]),1, axis =1)  
     
        for ep in range(self.epoch): 
            for j in range(len(self.y_train)):
        
                z = (w).dot(np.transpose(self.x_train[j]))
                
                #Activation Function
                if z < 0:
                    yhat = -1
                else:
                    yhat = 1
                    
                dw = self.eta*(self.y_train[j] - yhat)*self.x_train[j] 
                
                w += (dw)
            
            print("Epoch: ", ep, " / ", self.epoch)
        #Return weights
        return np.array(w)
