import numpy as np


from utils import *

DATA_DIR = 'cifar-10-batches-py'
DATA_BATCH = 'data_batch_1'
LABEL_MAP_FILE = 'batches.meta'


batch_path = '{}/{}'.format(DATA_DIR, DATA_BATCH)
label_map_path = '{}/{}'.format(DATA_DIR, LABEL_MAP_FILE)

input_dict = unpickle(batch_path)
label_names = unpickle(label_map_path)


for x, y in zip(input_dict[b'data'][:3], input_dict[b'labels'][:3]):

    #show_np_image(x, label_names[b'label_names'][y])


