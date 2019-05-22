#building FEED-FORWARD NEURAL NETWORK from scratch using numpy
#FFNN are the simplest of ANN devised till date,

#In a feed forward network information always moves one direction; it never goes backwards or form cycles (like in RNN).
#Feed Forward NN and Recurrent NN are types of Neural Nets, not types of Training Algorithms. 
#Training Algorithms are BackProp , Gradient Descent , etc which are used to train the networks. 
#In FFNN, the output of one layer does not affect itself whereas in RNN it does.

#PREDICT NEXT ONE IN SERIES - RNN

import numpy as np

#sigmoid function (aka nonlinearity function) :
#converts an input number into probability - a value between [0,1]
#Each single node will constitute of sigmoid function

def nonlin(x, deriv=False):
    if deriv==True:
        return (x*(1-x))
    return (1/(1+np.exp(-x)))
    
#Here nonlin returns either derivative or actual value of sigmoid function.
#Forward propogation : Sigmoid function is used
#Backward propogation : Derivative of sigmoid function is useful for calculating ERROR

#INPUT DATA
x = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]])

#OUTPUT DATA    
y = np.array([
    [0],
    [1],
    [1],
    [0]])
    
#Seeding helps us start at same point everytime since we are dealing with lots of random values. 
#Hence good for debugging.
np.random.seed(1)

#WEIGHTS : (More like SYNAPSES) - Connection between each NODE in nn
#let's build two synapse matrices
syn0 = 2*np.random.random((3,4)) - 1    #3x4 matrix of weights
syn1 = 2*np.random.random((4,1)) - 1    #4x1 matrix of weights

#Here '1' is a BIAS, we can set it to whatever we want and fine tune it relative to data and requirements
#BIAS is a must for building nn, so we simply  put it as 1
#DIMENSIONS OF THE SYNAPSE MATRIX (or Weight matrix) correspond to number of NODES and OUTPUTS
#ROW ----> n(NODES)
#COL ----> n(OUTPUTS)
#eg. syn0 = 2*np.random.random((nodes,outputs)) - 1

# Our Nodes' structure looks some what like this
#                      o       
#              
#              o    
#                      o       
#       
#  Input>>     o               o
#                      o
#   
#              o
#                      o   
    
    
#TRAINING
#--------
#While training we are continously inputting the data and updating the 'weights'overtime through backpropogation.
#It's like sculpting, every time you hit a hammer, you get close to what you want


#we use layers as an abstraction for NODES. We won't be coding individual neurons(or nodes)... 
#we are not making class for neuron.
#We are thinking it this way because that is the way the probabilities are calaculated

#more the range of loop is, better will be our model
for j in range(60000):
    
    #layers
    # each layer = sigmoid function([previous layer] x [synapse next to it])
    l0 = x
    l1 = nonlin(np.dot(l0, syn0))   
    l2 = nonlin(np.dot(l1, syn1))
    
    #BACKPROPGATION
    l2_error = y - l2
    
    #printing status
    if(j%10000)==0:
        print('Error : ' + str( np.mean(np.abs(l2_error))) )

    #calculte deltas
    l2_delta = l2_error*nonlin(l2, deriv=True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error*nonlin(l1, deriv=True)
    
    #update our synapses
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print('Output after training')
print(l2)


#Here model is : syn1 and syn0 (like coeeficient parameters in linear regression)
#even the abstracted nodes(layers) depend on synapse weights





 
