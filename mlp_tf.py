import numpy as np
from random import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
# array([0.1,0.2],[0.2,0.2])
# array([0.3],[0.4])


def generate_dataset(num_samples, test_size):
    x = np.array([[random() /2 for _ in range(2)] for _ in range(num_samples)]) #array( [[0.1, 0.2],[0.3. 0.4]])
    y = np.array([[i[0] + i[1]] for i in x]) #array( [[0.3],[0.7]])
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size) #30% of dataset is test set
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = generate_dataset(5000, 0.3)
    # print("x_test: \n {}".format(x_test))
    # print("y_test: \n {}".format(y_test))


# split set into training set and test set to evaluate how well the model does on the data it has never seen before
# to check whether the network has been able to generalize
# instead of doing from scratch we use scikitlearn library important for traditional ml
# build model
#keras is a high level libary on top of tensorflow- make tf code easy to code
# 2->5->1
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, input_dim = 2, activation = "sigmoid"),#fully connected layer
        tf.keras.layers.Dense(1, activation = "sigmoid")#fully connected layer
    ]) #sequential network - input goes from left to right

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

# compile model
model.compile(optimizer = optimizer, loss ="MSE")
# train model
model.fit(x_train,y_train,epochs =100,batch_size=1)
# evaluate model - how well by using the test set
print("\nModel Evaluation:")
model.evaluate(x_test,y_test,verbose = 1)
# make predictions
data =  np.array([[0.1,0.2],[0.2,0.2]])
predictions = model.predict(data)

print("\nPredictions: ")
for d,p in zip(data,predictions):
    print("{} + {} =  {}".format(d[0],d[1],p[0]))