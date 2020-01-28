import pickle
import h5py
import numpy as np
from tqdm import tqdm
from utils import normalize_image

CIFAR10_DATA_PATH  = './cifar-10-python/cifar-10-batches-py/'
CIFAR10_DATA_FILES = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
NUM_BATCHES    = len(CIFAR10_DATA_FILES)
BATCH_SIZE     = 10000
NUM_TRAIN      = NUM_BATCHES * BATCH_SIZE
IMAGE_SIZE     = (32, 32)
IMAGE_PIXELS   = IMAGE_SIZE[0] * IMAGE_SIZE[1]
IMAGE_CHANNELS = 3

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
        f.close()
        return dict

class CIFAR10DataLoader():
    def __init__(self):
        pass

    def preprocess(self, files=CIFAR10_DATA_FILES):
        with h5py.File('cifar10.h5', 'w') as h5_file:
            data   = np.zeros(shape=(NUM_TRAIN, *IMAGE_SIZE, IMAGE_CHANNELS), dtype=np.float32)
            labels = np.zeros(shape=(NUM_TRAIN,), dtype=int)
            for i, file in enumerate(tqdm(files)):
                file_path = f'{CIFAR10_DATA_PATH}{file}'
                batch = unpickle(file_path)
                data_batch   = batch[b'data']
                labels_batch = batch[b'labels']

                data  [i*BATCH_SIZE:(i+1)*BATCH_SIZE] = np.moveaxis(
                    np.reshape(normalize_image(data_batch), (BATCH_SIZE, IMAGE_CHANNELS, *IMAGE_SIZE)),
                    1, -1
                )
                labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = labels_batch

            h5_file.create_dataset('data', data=data)
            h5_file.create_dataset('labels', data=labels)
            h5_file.close()