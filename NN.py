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


test_input = np.array([[0.2,0.1,-4],[0.5,0.2,-1],[0.1,0.3,0.8]]).T
test_labels = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]).T

batch_size = test_labels.size

layer1 = NN_layer(3, 10)

output_layer = NN_layer(10, 4, is_output_layer=True)

NN_layers = [layer1, output_layer]
log_loss = CrossEntropyLoss()


training_iterations = 4000
learning_rate = 0.05


print("==== Training ====")

for epoch in range(training_iterations):
    predictions = forward_pass(NN_layers, test_input, test_labels, log_loss)
    da_output_layer = predictions - test_labels
    backpropagation(NN_layers, da_output_layer)
    update_parameters(NN_layers, learning_rate)
    # TODO: Add regularization






# for epoch in zip(input_dict[b'data'][:3], input_dict[b'labels'][:3]):
#
#     pass#show_np_image(x, label_names[b'label_names'][y])


