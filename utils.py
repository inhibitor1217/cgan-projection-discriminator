import math
import numpy as np
import matplotlib.pyplot as plt

def normalize_image(image):
    return image.astype(np.float32) / 127.5 - 1.

IMAGE_OUTPUT_COLS = 16
def format_tensor(x, color=False):
    if color:
        if len(x.shape) == 4:
            (num_samples, height, width, channels) = x.shape
            rows = math.ceil(num_samples / IMAGE_OUTPUT_COLS)
            cols = min(num_samples, IMAGE_OUTPUT_COLS)
            out = np.zeros(shape=(height * rows, width * cols, channels))
            for row in range(rows):
                for col in range(cols):
                    out[row * height:(row + 1) * height, col * width:(col + 1) * width] = x[row * cols + col]
            return out
        elif len(x.shape) == 3:
            return x
    else:
        if len(x.shape) == 4:
            (num_samples, height, width, _) = x.shape
            rows = math.ceil(num_samples / IMAGE_OUTPUT_COLS)
            cols = min(num_samples, IMAGE_OUTPUT_COLS)
            out = np.zeros(shape=(height * rows, width * cols))
            for row in range(rows):
                for col in range(cols):
                    out[row*height:(row+1)*height, col*width:(col+1)*width] = x[row*cols+col, :, :, 0]
            return out
        elif len(x.shape) == 3:
            return x[:, :, 0]
        else:
            return x

def imshow(x, color=False):
    plt.imshow(format_tensor(x, color))