import numpy as np


def apply_reLU(input):
    # Apply Rectified Linear Unit activation function.
    a = np.maximum(input, 0)
    return a


def derivative_of_reLU(z):
    # Apply Rectified Linear Unit activation function.
    result = np.zeros(z.shape)
    result[z > 0] = 1
    return result


def softmax(x):
    # Apply column-wise Softmax to an array
    x=x.astype(float)
    if x.ndim==1:
        S=np.sum(np.exp(x))
        return np.exp(x)/S
    elif x.ndim==2:
        result=np.zeros_like(x)
        M,N=x.shape
        for n in range(N):
            S=np.sum(np.exp(x[:,n]))
            result[:,n]=np.exp(x[:,n])/S
        return result
    else:
        print("The input array is not 1- or 2-dimensional.")



class NN_layer():
    def __init__(self, input_size, layer_size, is_output_layer=False):
        # Use standard normal distribution to initialise the weights and biases.
        # And apply the vanishing gradient prevention trick [ *np.sqrt(2/input_size) ].
        self.w = np.random.randn(layer_size, input_size)*np.sqrt(2/input_size)
        self.b = (np.random.randn(layer_size)*np.sqrt(2/input_size)).reshape(layer_size,1)
        self.z = np.zeros(layer_size)  # This value is useful for backprop after forward pass.

        self.dz = np.zeros(layer_size)
        self.da = np.zeros(layer_size)
        self.dw = np.zeros(layer_size)
        self.db = np.zeros(layer_size)


        self.is_output_layer = is_output_layer  # If True - make turns the layer output into logits.


    def calculate_layer_activations(self, input, labels=None):
        '''
        Return the output values of the current layer, given an input
        :param input: activations of the previous layer [x].
        :return:
        '''
        self.z = np.add(self.w.dot(input), self.b) # This matrix will also be useful in backpropagation.
        a = apply_reLU(self.z)

        if self.is_output_layer:
            # Apply softmax for the last layer
            a = softmax(a)

        return a


    def calculate_gradients(self, propagated_da):
        self.da = propagated_da
        self.dz = np.multiply(self.da, derivative_of_reLU(self.z))
        m = self.da.shape[1]  # number of samples in the current batch
        self.db = 1/m*np.sum(self.dz, axis=1, keepdims=True)

        da_L_minus_1 = self.w.T.dot(self.dz)

        self.dw = 1/m*self.dz.dot(da_L_minus_1.T)

        return da_L_minus_1

    def shape(self):
        return self.w.shape


class CrossEntropyLoss():

    def __init__(self):
        self.loss = 0
        self.cost = 0

    def calculate_loss(self, input, labels):
        # Logarithmic loss
        self.loss = np.where(labels == 1, -np.log(input), -np.log(1 - input))
        # Calculate the cost of the current sample batch
        self.cost = np.sum(self.loss)/input.shape[1]


def forward_pass(NN_layers, input, labels, loss):
    activation = None
    for idx, layer in enumerate(NN_layers):
        if idx < 1:
            # First layer
            activation = layer.calculate_layer_activations(input)
        elif (idx < len(NN_layers)-1):
            # Hidden layers
            activation = layer.calculate_layer_activations(activation)
        else:
            # Ouput layer
            predictions = layer.calculate_layer_activations(activation)
            loss.calculate_loss(predictions, labels)
            print("\nPredictions", predictions)
            print("\nCost:", loss.cost)
            return predictions


def backpropagation(NN_layers, da_output_layer):
    da = None
    for idx, layer in enumerate(NN_layers[::-1]):
        if idx < 1:
            # Lastlayer
            da = layer.calculate_gradients(da_output_layer)
        else:
            # Hidden layers
            da = layer.calculate_gradients(da)



def update_parameters(NN_layers, learning_rate, L2reg_constant, m):
    for layer in NN_layers:
        layer.w = (1-learning_rate*L2reg_constant/m)*layer.w - learning_rate * layer.dw
        layer.b = layer.b - learning_rate * layer.db



def calculate_accuracy(predictions, labels):
    predicted_categories = np.argmax(predictions, axis=0)
    accuracy = np.sum(predicted_categories==labels)/len(labels)*100
    print("Accuracy: {}%".format(accuracy))




