import numpy as np
from tensorflow.keras.layers import Concatenate, Reshape, Conv2D, Conv2DTranspose, \
    BatchNormalization, Dense, GlobalAveragePooling2D, Dot, Add, Lambda, LeakyReLU, Activation
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from cifar10_data_loader import CIFAR10DataLoader, NUM_TRAIN, IMAGE_SIZE, IMAGE_CHANNELS, NUM_CLASSES
from utils import imshow

NOISE_INPUT_SHAPE = (1, 1, 128)

def define_generator():
    _input = Input(shape=NOISE_INPUT_SHAPE)
    _label = Input(shape=(NUM_CLASSES,))

    label = Reshape((1, 1, NUM_CLASSES))(_label)
    x = Concatenate()([_input, label])
    x = Conv2DTranspose(64, 4, 2, padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(64, 4, 2, padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(32, 4, 2, padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(16, 4, 2, padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(16, 4, 2, padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(3, 3, 1, activation='tanh', padding='same')(x)

    model = Model([_input, _label], x)
    # model.summary()
    return model

def define_discriminator():
    x = _input = Input(shape=(*IMAGE_SIZE, IMAGE_CHANNELS))
    _label = Input(shape=(NUM_CLASSES,))

    x = Conv2D(64, 4, 2, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(64, 4, 2, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(128, 4, 2, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(128, 4, 2, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(NUM_CLASSES, 3, 1, activation='relu', padding='same')(x)
    x = GlobalAveragePooling2D()(x)

    linear = Dense(1)(x)
    dot = Dot(axes=-1)([x, _label])
    x = Add()([linear, dot])
    x = Activation('sigmoid')(x)

    model = Model([_input, _label], x)
    # model.summary()
    return model

if __name__ == '__main__':

    data_loader = CIFAR10DataLoader()

    g = define_generator()
    d = define_discriminator()

    def sample_image(num_samples=16):
        noise = np.random.normal(size=(num_samples * NUM_CLASSES, *NOISE_INPUT_SHAPE))
        label = np.zeros(shape=(num_samples * NUM_CLASSES, NUM_CLASSES), dtype=np.float32)
        for i in range(NUM_CLASSES):
            label[np.arange(i*num_samples, (i+1)*num_samples), i] = 1.
        sampled = g.predict([noise, label])
        imshow(sampled, color=True)

    _input_image = Input(shape=(*IMAGE_SIZE, IMAGE_CHANNELS))
    _input_noise = Input(shape=NOISE_INPUT_SHAPE)
    _input_label = Input(shape=(NUM_CLASSES,))

    real_prediction = d([_input_image, _input_label])
    fake_prediction = d([g([_input_noise, _input_label]), _input_label])

    d_trainer_real = Model([_input_image, _input_label], real_prediction)
    d_trainer_fake = Model([_input_noise, _input_label], fake_prediction)
    g_trainer      = Model([_input_noise, _input_label], fake_prediction)

    g.trainable = True
    d.trainable = False
    g_trainer.compile(optimizer=Adam(lr=1e-4, beta_1=.9, clipvalue=10, clipnorm=10), loss='binary_crossentropy')

    g.trainable = False
    d.trainable = True
    d_trainer_real.compile(optimizer=Adam(lr=1e-4, beta_1=.9, clipvalue=10, clipnorm=10), loss='binary_crossentropy')
    d_trainer_fake.compile(optimizer=Adam(lr=1e-4, beta_1=.9, clipvalue=10, clipnorm=10), loss='binary_crossentropy')

    # g_trainer.summary()
    # d_trainer_real.summary()
    # d_trainer_fake.summary()

    d_real_losses = []
    d_fake_losses = []
    g_losses = []

    epochs = 2000
    batch_size = 1024
    save_freq = 2000

    for epoch in range(epochs):
        data_gen, steps = data_loader.batch_generator(batch_size=batch_size)

        epoch_d_loss_real = 0.
        epoch_d_loss_fake = 0.
        epoch_g_loss      = 0.

        for step in range(steps):
            image, label = next(data_gen)
            cur_batch_size = image.shape[0]
            noise = np.random.normal(0, 1, size=(cur_batch_size, *NOISE_INPUT_SHAPE))

            y_real = np.random.uniform(0.9, 1.0, size=(cur_batch_size, 1))
            y_fake = np.random.uniform(0.0, 0.1, size=(cur_batch_size, 1))

            d_loss_real = d_trainer_real.train_on_batch([image, label], y_real)
            d_loss_fake = d_trainer_fake.train_on_batch([noise, label], y_fake)
            g_loss      = g_trainer.train_on_batch([noise, label], y_real)

            epoch_d_loss_real += d_loss_real * cur_batch_size
            epoch_d_loss_fake += d_loss_fake * cur_batch_size
            epoch_g_loss      += g_loss      * cur_batch_size

            print(f'epoch={epoch}, step={step}, d_real={d_loss_real}, d_fake={d_loss_fake}, g_loss={g_loss}')

        epoch_d_loss_real /= NUM_TRAIN
        epoch_d_loss_fake /= NUM_TRAIN
        epoch_g_loss      /= NUM_TRAIN

        d_real_losses.append(epoch_d_loss_real)
        d_fake_losses.append(epoch_d_loss_fake)
        g_losses.append(epoch_g_loss)

        if (epoch+1) % save_freq == 0:
            fig = plt.figure(figsize=(16, 16))
            sample_image()
            fig.savefig(f'{epoch+1}.png')

    plt.figure(figsize=(8, 8))
    plt.plot(d_real_losses)
    plt.plot(d_fake_losses)
    plt.plot(g_losses)
    plt.legend(['d_real', 'd_fake', 'g'])
    plt.show()