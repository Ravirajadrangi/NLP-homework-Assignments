import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
    number_examples = len(data)
    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    z1 = np.dot(data, W1) + b1
    h = sigmoid(z1) # element wise sigmoid function=
    z2 = np.dot(h, W2) + b2
    yhat = softmax(z2) # element wise softmax
    # Cost Function
    cost = np.sum(labels*np.log(yhat))
    #print('yhat', yhat)
    #print(z2)
    # Backward Propagation
    delta_1 = yhat-labels
    delta_2 = np.dot(delta_1, np.transpose(W2))* sigmoid_grad(z1)
    #print('LABELS\n', labels)
    print("ERROR 1", delta_1)
    gradW2 = np.dot(np.transpose(h),delta_1)
    gradb2 = delta_1
    gradW1 = np.dot(delta_2, np.transpose(W1))
    gradb1 = delta_2

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    print(gradW2)
    return cost, grad



def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print('Running basic tests....')

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum, aka each row is  an input vector
    #print(data)
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )
    #print(params.shape)
    #print(params)

    forward_backward_prop(data, labels, params,
                          dimensions)
'''
    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
       dimensions), params)
'''
def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":

    sanity_check()
    your_sanity_checks()