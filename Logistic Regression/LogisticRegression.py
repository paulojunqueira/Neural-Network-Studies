import numpy as np
import matplotlib.pyplot as plt
import os

from skimage.io import imread_collection, imshow
from skimage.transform import resize,downscale_local_mean

from sklearn.metrics import classification_report

def sigmoid(z):
    return 1/(1+np.exp(-z))


#your path 
cats_dir = 'data\\data\\training_set\\cats\\*.jpg'
notcats_dir = 'data\\data\\training_set\\notcats\\*.jpg'


tcats_dir = 'data\\data\\test_set\\cats\\*.jpg'
tnotcats_dir = 'data\\data\\test_set\\notcats\\*.jpg'

#Dataset Cats vs dog

#Train Data path 
cats = 'data\\data\\train\\cat.*.jpg'
dogs = 'data\\data\\train\\dog.*.jpg'


def formatData(size, cats_dir, dogs_dir, dim):
    """ This function reads dataset, separeta a slice (size),
        resize photos (dim) and create labels (0, 1) for respective
        classes. It randomizes positions.
    """
    train_cats = imread_collection(cats_dir)
    train_cats = train_cats[0:size]
    train_cats = np.array(list(map(lambda x: resize(x, (dim,dim,3)),train_cats)))
    y_cats = np.ones((1,len(train_cats)))

    train_dogs = imread_collection(dogs_dir)
    train_dogs = train_dogs[0:size]
    train_dogs = np.array(list(map(lambda x: resize(x, (dim,dim,3)),train_dogs)))
    y_notcats = np.zeros((1,len(train_dogs)))

    X_train = np.concatenate([train_cats, train_dogs])
    X_train.shape
    Y_train = np.concatenate([y_cats, y_notcats], axis = 1)
    Y_train.shape

    sample = np.random.choice(range(2*size),2*size, replace = False )
    X_train = X_train[sample]
    Y_train = Y_train[0,sample]

    return X_train, Y_train


X_train, Y_train = formatData(800, cats, dogs, 64)
X_test, Y_test = formatData(50, tcats_dir, tnotcats_dir)

X_tr = X_train.reshape(X_train.shape[0],-1).T
X_tes = X_test.reshape(X_test.shape[0],-1).T


imshow(X_train[398])

def logisticRegression(X,Y,maxIterations,eta, verbose = True):
    
    size,m = X.shape
    cost = []
    w = np.zeros((size,1))
    b = 0
    alpha = 0.025
    costi = 1.0 + alpha
    i = 0
    while i < maxIterations or costi<alpha:
    
        if verbose:
            print(f'Epoch:{i+1}/{maxIterations}, Loss: {costi}')
        #Feed Forword 
        Z = sigmoid(np.dot(w.T,X)+b)               
        costi = (-1/m)*np.sum(((Y*np.log(Z)) + (1-Y)*np.log(1-Z)), axis = 1)   
        cost.append(costi)

        #BackPropagation
        dw = (1/m)*(np.dot(X,((Z-Y).T)))
        db = (1/m)*np.sum((Z-Y), axis = 1)
        #Update
        w = w-eta*dw
        b = b-eta*db

        i+=1
    return w, b, cost 


def pred(w,b, X):
    size, m = X.shape
    yhat = np.zeros((1,m))

    #Feed Forword
    Z = sigmoid(np.dot(w.T,X)+b)  
    for i in range(m):

        if Z[0,i] <= 0.5:
            yhat[0,i] = 0
        else:
            yhat[0,i] = 1

    return yhat

def acc(yhat,ytrue):
    return 100 - np.mean(np.abs(yhat - ytrue)) * 100

w,b,cost = logisticRegression(X_tr,Y_train, 50000,0.002)
plt.plot(cost)
yhat_train = pred(w,b,X_tr)

acc_tr = 100 - np.mean(np.abs(yhat_train - Y_train)) * 100
print(acc_tr)

yhat_test = pred(w,b,X_tes)
acc_tes =  100 - np.mean(np.abs(yhat_test - Y_test)) * 100
print(acc_tes)



dim = 64
gatos = imread_collection('data\\data\\eval\\*.jpeg')
gatos = np.array(list(map(lambda x: resize(x, (dim,dim,3)),gatos)))
gatos = gatos.reshape(gatos.shape[0],-1).T

result = pred(w,b,gatos)
list(map(lambda x: 'Cat' if x == 1 else 'Dog', result[0]))



# pantera = resize(pantera[0], (64,64,3))
# imshow(pantera)
# xPantera = pantera.reshape(pantera.shape[0]*pantera.shape[1]*pantera.shape[2],1)

# yhat = pred(w,b,xPantera)

import json
data = {'w':w, 'b': b}

with open('data.txt', 'w') as file:
    json.dump(data, file)


with open('filename.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('data.pickle', 'rb') as handle:
    b = pickle.load(handle)