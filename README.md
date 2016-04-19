# Overview

This code uses TensorFlow to build a NN classifier with one hidden layer that can determine the relevance of pages. The code is written for Python3.

You can run in command line:

```
python3 run_classify.py

```

To get the best performance, several parameters need to be tuned in nn_classifier.py, which are:

```
batch_size: number of point used in calculating gradient
dropout_p: dropout rate
decay_rate: decay rate for learning rate in SGD
num_steps: number of iterations for SGD

```

# Behind the code

### Feature generation:

1. Extract text from body of html
2. Remove stopwords
3. Define keywords: words with high frequency in **positive pages**, but not with high frequency in **negative pages**.
4. For each sample (page), form features by counting frequency of each keyword in the page.

### Modeling:

A simple neural network is used with one hidden layer. Rectifier function Relu() is selected as the activation function for the hidden layer. The number of unit for this hidden layer is arbitrarily set to 1024 (this parameter can be tuned based on the accuracy of valid dataset). Cross entropy is used as loss function, and dropout (p = 0.8) is applied to avoid overfitting. The weights and biases are optimizied using stochastic gradient descent (batch_size = 128) with exponential decaied learning rate.

