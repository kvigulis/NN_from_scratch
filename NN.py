from utils import *
from NN_utils import *


DATA_DIR = 'cifar-10-batches-py'
DATA_BATCH = 'data_batch_1'
LABEL_MAP_FILE = 'batches.meta'


batch_path = '{}/{}'.format(DATA_DIR, DATA_BATCH)
label_map_path = '{}/{}'.format(DATA_DIR, LABEL_MAP_FILE)

input_dict = unpickle(batch_path)
label_names = unpickle(label_map_path)

# Keep the same seed for DEBUG
np.random.seed(1)


# test_input = np.array([[0.2,0.1,-4],[0.5,0.2,-1],[0.1,0.3,0.8]]).T
# test_labels = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]).T

labels_list = input_dict[b'labels']
labels = np.zeros((len(labels_list[:]), 10))
labels[np.arange(len(labels_list[:])), labels_list] = 1

layer1 = NN_layer(3072, 64)
layer2 = NN_layer(64, 64)
output_layer = NN_layer(64, 10, is_output_layer=True)

NN_layers = [layer1, layer2, output_layer]
log_loss = CrossEntropyLoss()


training_iterations = 4000
learning_rate = 0.05
L2reg_constant = 0.005


print("==== Training ====")

for epoch in range(training_iterations):
    predictions = forward_pass(NN_layers, input_dict[b'data'].T/255, labels.T, log_loss)
    calculate_accuracy(predictions, input_dict[b'labels']) # Print prediction accuracy
    da_output_layer = predictions - labels.T
    backpropagation(NN_layers, da_output_layer)
    update_parameters(NN_layers, learning_rate, L2reg_constant, labels.shape[1])
    # TODO: Add regularization





