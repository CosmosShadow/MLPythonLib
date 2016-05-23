
"""
A module to implement the stochastic gradient descent 
learning algorithm for softmaxRegression.
"""

import random
import numpy as np

class softMax(object):

    def __init__(self, lenX, lenY):
        self.weights = (0.5 - np.random.random((lenY, lenX))) * 0.12

    def feedforward(self, x):
        """Return the output of the prob as ``x`` is input."""
        a = softmax(self.weights, x)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, eta_decay,
            test_data = None):
        """Train the softmax regression using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        regression will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: 
            n_test = len(test_data)
        n = len(training_data)
        threshold = 10
        accuracy = []
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                correct = self.evaluate(training_data, convert = True)
                accuracy.append(correct)
                print "Epoch {0}: {1} / {2} CostTrain: {3}".format(
                    j, self.evaluate(test_data), n_test, self.costValue(training_data))  
                if j > threshold:
                    if np.mean(accuracy[-5:]) - np.mean(accuracy[-10:-5]) <= 0:
                        eta = eta / eta_decay
                        threshold = j + 20
                        print eta                 
            else:
                print "Epoch {0} complete".format(j)
        return self.weights

    def update_mini_batch(self, mini_batch, eta):
        """Update the softmax's weights by applying
        gradient descent to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        for i in range(len(mini_batch)):
            x, y = mini_batch[i]
            if i == 0:
                xset = x
                yset = y
            else:
                xset = np.hstack((xset, x))
                yset = np.hstack((yset, y))

        delta_nabla_w = self.gradients(xset, yset)            
        self.weights = self.weights + (eta / len(mini_batch)) * delta_nabla_w
                        
    def gradients(self, x, y):
        """Return nabla_w representing the
        gradient for the cost function C_x."""
        linearcombi = np.dot(self.weights, x) 
        explinearcombi = np.exp(linearcombi)
        outporb = explinearcombi / explinearcombi.sum(axis = 0, keepdims = True)
         
        dist = y - outporb 
        
        dw = np.zeros(self.weights.shape)
        for i in xrange(x.shape[1]):
            dw += np.outer(dist[:, i], x[:, i])
        nabla_w = dw / x.shape[1]
        
        return nabla_w
    
    def costValue(self, dataset):
        '''return value of the cost function after every epoch'''
        likelihood = 0
        l = len(dataset)
        for x,y  in dataset:
            lc = np.dot(self.weights, x)
            explc = np.exp(lc)
            outputprob = explc / explc.sum(axis = 0, keepdims = True)
            ind = np.argmax(y)
            likelihood += np.log(outputprob[ind])
        cost = -likelihood / l
        return cost

    def evaluate(self, dataset, convert = False):
        """Return the number of test inputs for which the softmax
        regression outputs the correct result. Note that the softmax
        regression's output is assumed to be the index of whichever
        output in the final layer has the highest activation."""
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in dataset]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in dataset]
        return sum(int(x == y) for (x, y) in results)
     
#softmax function
def softmax(W, x):
   """the softmax function"""
   vec = np.dot(W, x)
   vec1 = np.exp(vec)
   outprob = vec1.T / np.sum(vec1)
   return outprob.T

   