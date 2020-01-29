import pickle
import h5py
import numpy as np
from tqdm import tqdm
from utils import normalize_image
import os
from datetime import datetime
import math

CIFAR10_DATA_PATH  = './cifar-10-python/cifar-10-batches-py/'
CIFAR10_DATA_FILES = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
H5_FILE            = './cifar10.h5'
DATASET_X          = 'data'
DATASET_Y          = 'labels'
NUM_BATCHES    = len(CIFAR10_DATA_FILES)
BATCH_SIZE     = 10000
NUM_TRAIN      = NUM_BATCHES * BATCH_SIZE
IMAGE_SIZE     = (32, 32)
IMAGE_PIXELS   = IMAGE_SIZE[0] * IMAGE_SIZE[1]
IMAGE_CHANNELS = 3
NUM_CLASSES    = 10

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
        f.close()
        return dict

class CIFAR10DataLoader():
    def __init__(self):
        pass

    def preprocess(self, files=CIFAR10_DATA_FILES):
        with h5py.File(H5_FILE, 'w') as h5_file:
            data   = np.zeros(shape=(NUM_TRAIN, *IMAGE_SIZE, IMAGE_CHANNELS), dtype=np.float32)
            labels = np.zeros(shape=(NUM_TRAIN, NUM_CLASSES), dtype=int)
            for i, file in enumerate(tqdm(files)):
                file_path = f'{CIFAR10_DATA_PATH}{file}'
                batch = unpickle(file_path)
                data_batch   = batch[b'data']
                labels_batch = batch[b'labels']

                data  [i*BATCH_SIZE:(i+1)*BATCH_SIZE] = np.moveaxis(
                    np.reshape(normalize_image(data_batch), (BATCH_SIZE, IMAGE_CHANNELS, *IMAGE_SIZE)),
                    1, -1
                )
                labels[np.arange(i*BATCH_SIZE, (i+1)*BATCH_SIZE), labels_batch] = 1.

            h5_file.create_dataset(DATASET_X, data=data)
            h5_file.create_dataset(DATASET_Y, data=labels)
            h5_file.close()

    def generator(self, shuffle=True):
        def _generator(shuffle):
            with h5py.File(H5_FILE, 'r') as h5_file:
                seed = (id(self) + os.getpid() + int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
                np.random.seed(seed)
                indices = np.arange(h5_file[DATASET_X].shape[0])
                if shuffle:
                    np.random.shuffle(indices)
                for idx in indices:
                    yield h5_file[DATASET_X][idx], h5_file[DATASET_Y][idx]
                h5_file.close()
        return _generator(shuffle), NUM_TRAIN

    def batch_generator(self, batch_size, shuffle=True, remainder=True):
        def _generator(shuffle):
            single_gen, _ = self.generator(shuffle)
            batch_x = []
            batch_y = []
            for x, y in single_gen:
                batch_x.append(x)
                batch_y.append(y)
                if len(batch_x) == batch_size:
                    yield np.array(batch_x), np.array(batch_y)
                    del batch_x[:]
                    del batch_y[:]
            if remainder and len(batch_x) > 0:
                yield np.array(batch_x), np.array(batch_y)
        steps = math.ceil(NUM_TRAIN / batch_size) if remainder else (NUM_TRAIN // batch_size)
        return _generator(shuffle), steps