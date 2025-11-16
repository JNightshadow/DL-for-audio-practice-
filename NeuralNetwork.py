import numpy as np
from random import random
# save the activations and derivatives
#implement backpropagation
#implement gradient descent 
#implement a train
#train our network with some dummy dataset!

class MLP:
    #Represemtation of a Multi-Layer Perceptron Neural Network
    def __init__(self,num_inputs = 3,hidden_layers = [3,5],num_outputs = 2):
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.hidden_layers + [self.num_outputs]
        
        #initialize weights
        self.weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i],layers[i+1])  #Initialize weights with random values
            self.weights.append(w)
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i]) #array of zeroes with length of number of neurons in each layer
            activations.append(a)
        self.activations = activations

        derivatives = []
        for i in range(len(layers)-1):
            d = np.zeros((layers[i],layers[i+1])) #array of zeroes with length of number of neurons in each layer
            derivatives.append(d)
        self.derivatives = derivatives


    def forward_propagate(self,inputs):
        activations = inputs #for first layer the activations are the input itself.
        self.activations[0] = inputs
        for i,w in enumerate(self.weights):
            #calculate the net inputs - matrix multiplication with activation of previous layer with weight matrix
            net_inputs = np.dot(activations,w)
            #calculate the activations
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations 
            # a_3 =sigmoid(h_3)
            # h_3 = a_2 * W_2
        return activations
    
    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    #Pass the error
    def back_propagate(self,error,verbose = False):

        #We are propagating error from the output layer towards the left. Hence reversed!
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations) # ndarray([0.1, 0.2] -> [[0.1,0.2]])
            delta_reshaped = delta.reshape(delta.shape[0],-1).T #Transpose the array
            #Rearranging to form a vertical array - vertical vector
            current_activations = self.activations[i] # ndarray([0.1, 0.2] -> [[0.1],[0.2]])
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0],-1)
            self.derivatives[i] = np.dot(current_activations_reshaped,delta_reshaped)
            error = np.dot(delta,self.weights[i].T)
        
            if verbose:
                print("Derivatives for W{}: {}".format(i,self.derivatives[i]))
        return error
    
    def gradient_descent(self,learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]

            derivatives = self.derivatives[i]


            #update weights
            weights += derivatives * learning_rate

    
    def _sigmoid_derivative(self, x):
        return x *(1.0-x)
    
    def train(self,inputs,targets,epochs,learning_rate):
        
        for i in range(epochs):
            sum_error = 0
            for input,target in zip(inputs,targets): #getting inputs and targets with index. zip packs the two inputs :)
                 #forward propagation
                output = self.forward_propagate(input)
                #calculate error
                error = target - output
                #backprop
                self.back_propagate(error)
                self.gradient_descent(learning_rate)

                sum_error += self._mse(target,output) #mean squared error
            #report error
            print("Error:  {} at epoch {}".format(sum_error/len(inputs),i))

    def _mse(self,target,output):
        return np.average((target - output)**2)

if __name__=="__main__":

    inputs = np.array([[random() /2 for _ in range(2)] for _ in range(100)]) #array( [[0.1, 0.2],[0.3. 0.4]])



    targets = np.array([[i[0] + i[1]] for i in inputs]) #array( [[0.3],[0.7]])
    #create an mlp
    mlp = MLP(2,[5],1)

    #train our mlp
    mlp.train(inputs,targets,50,0.1)

    #create dummy data
    input = np.array([0.3,0.1])
    target = np.array([0.4])
    output = mlp.forward_propagate(input)
   
    print()
    print()
    print("Our network believes that {} + {} is equal to {}".format(input[0],input[1],output))
