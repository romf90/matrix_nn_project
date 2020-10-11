# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 19:43:57 2020

@author: רום
"""
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
  return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:
  '''
  A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)

  *** DISCLAIMER ***:
  The code below is intended to be simple and educational, NOT optimal.
  Real neural net code looks nothing like this. DO NOT use this code.
  Instead, read/run it to understand how this specific network works.
  '''
  def __init__(self):
    
    self.inputSize = 2
    self.hiddenSize = 2
    self.outputSize = 1
    
    # Weights
    self.loss_history = []
    self.hidden_weights = np.random.uniform(size=( self.inputSize,self.hiddenSize))  
    self.output_weights = np.random.uniform(size=( self.hiddenSize, self.outputSize))
   
    # Biases
    
    self.hidden_bias = np.random.uniform(size=(1,self.hiddenSize))   
    self.output_bias = np.random.uniform(size=(1,self.outputSize))   
    
  def feedforward(self, x):
    # x is a numpy array with 2 elements.    
    hiden_layer_ff = sigmoid(np.dot(x,self.hidden_weights) + self.hidden_bias)
    output_ff = sigmoid(np.dot(hiden_layer_ff,self.output_weights) + self.output_bias)
    return  output_ff

  def train(self, data, all_y_trues):
    '''
    - data is a (n x 2) numpy array, n = # of samples in the dataset.
    - all_y_trues is a numpy array with n elements.
      Elements in all_y_trues correspond to those in data.
    '''
    learn_rate = 0.002
    epochs = 1000 # number of times to loop through the entire dataset

    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
          
        # --- Do a feedforward (we'll need these values later)
        
        hiden_layer_ff = sigmoid(np.dot(x,self.hidden_weights) + self.hidden_bias)
        output_ff = np.dot(hiden_layer_ff,self.output_weights) + self.output_bias      
        y_pred =  sigmoid(output_ff)

        # --- Calculate partial derivatives.        
        # --- Naming: d_L_d_w1 represents "partial L / partial w1"
        
        d_L_d_ypred = -2 * (y_true - y_pred)


        d_ypred_d_output_weights = hiden_layer_ff * deriv_sigmoid(output_ff)        
        d_ypred_d_output_bias = deriv_sigmoid(output_ff)
        d_ypred_d_hiden_layer_ff = self.output_weights * deriv_sigmoid(output_ff)                   
        deriv_sigmoid_matrix = np.vstack((deriv_sigmoid(np.dot(x,self.hidden_weights) + self.hidden_bias),
                                          deriv_sigmoid(np.dot(x,self.hidden_weights) + self.hidden_bias)))       
        d_hidden_d_b = deriv_sigmoid_matrix[0]                
        d_h_d_hidden_weights = deriv_sigmoid_matrix.T * x
       
        # --- Update weights and biases --- #
        
        self.output_weights -= d_ypred_d_output_weights.T * learn_rate * d_L_d_ypred      
        self.output_bias -= learn_rate * d_L_d_ypred * d_ypred_d_output_bias       
        self.hidden_weights -=d_h_d_hidden_weights.T * d_ypred_d_hiden_layer_ff  * learn_rate * d_L_d_ypred      
        self.hidden_bias -= d_ypred_d_hiden_layer_ff.T * learn_rate * d_L_d_ypred * d_hidden_d_b       
        
      # --- Calculate total loss at the end of each epoch
      y_preds = np.apply_along_axis(self.feedforward, 1, data)
      loss = mse_loss(all_y_trues,y_preds) - 0.25     
      self.loss_history.append(loss)
     
      if epoch % 10 == 0:   
        print("Epoch %d loss: %.3f" % (epoch, loss))
    fig, ax = plt.subplots()
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.plot(self.loss_history)
# Define dataset
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

# Train our neural network!
network = OurNeuralNetwork()
network.train(data, all_y_trues)
