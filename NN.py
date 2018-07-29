import numpy as np


from utils import *



DATA_DIR = 'cifar-10-batches-py'
DATA_BATCH = 'data_batch_1'
LABEL_MAP_FILE = 'batches.meta'


batch_path = '{}/{}'.format(DATA_DIR, DATA_BATCH)
label_map_path = '{}/{}'.format(DATA_DIR, LABEL_MAP_FILE)

input_dict = unpickle(batch_path)
label_names = unpickle(label_map_path)

# Keep the same seed for DEBUG
np.random.seed(1)

def apply_reLU(input):
    # Apply Rectified Linear Unit activation function.
    a = np.maximum(input, 0)
    return a


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
    def __init__(self, input_size, layer_size, softmax_output=False):
        # Use standard normal distribution to initialise the weights and biases.
        # And apply the vanishing gradient prevention trick [ *np.sqrt(2/input_size) ].
        self.w = np.random.randn(layer_size, input_size)*np.sqrt(2/input_size)
        self.b = (np.random.randn(layer_size)*np.sqrt(2/input_size)).reshape(layer_size,1)
        self.z = np.zeros(layer_size)  # Derivative of activation function with respect to z.
        self.da_dz = np.zeros(layer_size) # Derivative of activation function with respect to z.


        # Makes a logits layer
        self.softmax_output = softmax_output

    def calculate_layer_activations(self, input):
        '''
        Return the output values of the current layer, given an input
        :param input: activations of the previous layer [x].
        :return:
        '''

        print("w", self.w.shape)
        print("b", self.b.shape)
        print("input", input.shape)
        print("self.w.T.dot(input)", self.w.dot(input).shape)
        self.z = np.add(self.w.dot(input), self.b) # This matrix will also be useful in backpropagation.
        print("self.z", self.z.shape)
        a = apply_reLU(self.z)

        if self.softmax_output:
            a = softmax(a)

        return a



class CrossEntropyLoss():

    def __init__(self):
        self.loss = 0
        self.cost = 0

    def calculate_loss(self, input, labels):
        # Logarithmic loss
        self.loss = np.where(labels == 1, -np.log(input), -np.log(1 - input))
        print("loss input shape", input.shape[1])

        # Calculate the cost of the current sample batch
        self.cost = np.sum(self.loss)/input.shape[1]



class BackProp():

    def __int__(self):
        # Update the derivatives of the nodes.
        self.da_dz[a > 0] = 1
        print("da/dz", self.da_dz)



test_input = np.array([[2,1,4],[3,3,-1],[1,3,8]]).T

layer1 = NN_layer(3, 4)
output = NN_layer(4, 2, softmax_output=True)
log_loss = CrossEntropyLoss()


print(test_input)
print(layer1.w)
print(layer1.b)



activation_l1 = layer1.calculate_layer_activations(test_input)
activation_output = output.calculate_layer_activations(activation_l1)

log_loss.calculate_loss(activation_output, np.array([1,1,0]))



print("Output:", activation_output)


print("Loss", log_loss.loss)
print("Cost", log_loss.cost)


for x, y in zip(input_dict[b'data'][:3], input_dict[b'labels'][:3]):

    pass#show_np_image(x, label_names[b'label_names'][y])


