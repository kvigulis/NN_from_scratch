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


def derivative_of_reLU(z):
    # Apply Rectified Linear Unit activation function.
    result = np.zeros(z.shape)
    result[z > 0] = 1

    print("Derivative of ReLU ", result)
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


        # Makes a logits layer
        self.is_output_layer = is_output_layer


    def calculate_layer_activations(self, input, labels=None):
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





test_input = np.array([[2,1,-4],[3,3,-1],[1,3,8]]).T

test_labels = np.array([1,1,0])

batch_size = test_labels.size

layer1 = NN_layer(3, 4)
output_layer = NN_layer(4, 2, is_output_layer=True)


log_loss = CrossEntropyLoss()


print(test_input)
print(layer1.w)
print(layer1.b)



activation_l1 = layer1.calculate_layer_activations(test_input)
activation_output = output_layer.calculate_layer_activations(activation_l1)

log_loss.calculate_loss(activation_output, test_labels)


print("Output:", activation_output)


print("Loss", log_loss.loss)
print("Cost", log_loss.cost)

da_output_layer =  activation_output - test_labels

print("da_output_layer", da_output_layer)

da_layer1 = output_layer.calculate_gradients(da_output_layer)
layer1.calculate_gradients(da_layer1)
print("\n\n== Gradients da==")
print("ouput da", output_layer.da)
print("layer1 da", layer1.da)

print("\n\n== Gradients dw==")
print("ouput dw", output_layer.dw)
print("layer1 dw", layer1.dw)


print("\n\n== Gradients db==")
print("ouput db", output_layer.db)
print("layer1 db", layer1.db)



for x, y in zip(input_dict[b'data'][:3], input_dict[b'labels'][:3]):

    pass#show_np_image(x, label_names[b'label_names'][y])


