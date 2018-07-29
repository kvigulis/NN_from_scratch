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



class NN_layer():
    def __init__(self, input_size, layer_size):
        # Use standard normal distribution to initialise the weights and biases.
        # And apply the vanishing gradient prevention trick [ *np.sqrt(2/input_size) ].
        self.w = np.random.randn(layer_size, input_size)*np.sqrt(2/input_size)
        self.b = (np.random.randn(layer_size)*np.sqrt(2/input_size)).reshape(layer_size,1)
        self.da_dz = np.zeros(layer_size) # Derivative of activation function with respect to z.


    def calculate_layer_activations(self, input):
        '''
        Return the output values of the current layer, given an input
        :param input: activations of the previous layer [x].
        :return:
        '''

        print("w.x", self.w.dot(input.T))
        print("b", self.b)
        z = np.add(self.w.dot(input.T), self.b)


        a = apply_reLU(z)

        self.da_dz[a > 0] = 1
        print("a", a)
        print("da/dz", self.da_dz)

        return a



class Output_layer():
    # Network's output layer
    def __init__(self, number_of_classes):
        self.output_nodes = np.zeros(number_of_classes)

    def calculate_layer_activations(self, input):
        '''
        Return the output values of the current layer, given an input
        :param input: activations of the previous layer and the values of input layer
        :return:
        '''
        z = self.w.dot(input) + self.b
        a = apply_reLU(z)

        return a


test_input = np.array([[2,1,4],[3,6,-1]])
layer1 = NN_layer(3, 4)

output = NN_layer(4, 4)


print(test_input)
print(layer1.w)
print(layer1.b)



activation_l1 = layer1.calculate_layer_activations(test_input)
activation_output = output.calculate_layer_activations(activation_l1)



for x, y in zip(input_dict[b'data'][:3], input_dict[b'labels'][:3]):

    pass#show_np_image(x, label_names[b'label_names'][y])


