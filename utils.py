import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def show_np_image(np_array, label):
    # Convert the 3072 image vector to shape 32,32,3 and display with matplot.
    np_image = np.rot90(np.reshape(np_array, (32, 32, 3), order='F'), k=-1)
    plt.imshow(np_image)
    plt.title('Label={}'.format(label))
    plt.show()