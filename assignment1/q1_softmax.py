import numpy as np
import random


def softmax(x):

    """
    Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.
    You might find numpy functions np.exp, np.sum, np.reshape,
    np.max, and numpy broadcasting useful for this task. (numpy
    broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

    You should also make sure that your code works for one
    dimensional inputs (treat the vector as a row), you might find
    it helpful for your later problems.

    You must implement the optimization in problem 1(a) of the 
    written assignment!
    """

    ### YOUR CODE HERE
    #x -= np.amax(x)
    #result = np.exp(x) / np.sum(np.exp(x), axis=0)
    #print('result', result)
    dimensions = x.ndim
    print(dimensions)
    if x.ndim == 1:
        x -= np.max(x)
        p = np.exp(x) / np.sum(np.exp(x))
    else:
        max_list = np.max(x, axis=1)
        # subtract max values to make it numerically stable
        for i in range(len(max_list)):
            x[i] -= max_list[i]
        e = np.exp(x)
        #print(e)
        total = np.sum(np.exp(x), axis=1)
        p = np.zeros(np.shape(e))
        # divide each vector example by the sum of that example
        for i in range(len(e)):
            p[i] = e[i] / total[i]

        #print('soft', p)

        #p = np.exp(x) / np.sum(np.exp(x), axis=1)

    return p

def test_softmax_basic():
    """
    Some simple tests to get you started. 
    Warning: these are not exhaustive.
    """
    print("Running basic tests...")
    test1 = softmax(np.array([1,2]))
    print('Test 1', test1)
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print(test2)
    assert np.amax(np.fabs(test2 - np.array(
        [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6

    test3 = softmax(np.array([[-1001,-1002]]))
    print('Test 3',test3)
    assert np.amax(np.fabs(test3 - np.array(
        [0.73105858, 0.26894142]))) <= 1e-6

    print("You should verify these results!\n")

def test_softmax():
    print("Running your tests...")
    x = np.array([[1005, 1010,1012], [5000, 5004, 508], [12, 22, 14.5]])
    test1 = softmax(x)
    print(test1)
    print(test1[0])
    print(np.sum(test1[1]))
    print(np.sum(test1, axis=1))
    #assert np.amax(np.fabs(test1-np.array()))



if __name__ == "__main__":
    test_softmax_basic()
    #test_softmax()


#x = np.random.rand(3,5)
#softmax(x)