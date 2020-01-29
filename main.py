import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from cifar10_data_loader import CIFAR10DataLoader, IMAGE_SIZE, IMAGE_CHANNELS

def define_classifier():
    x = _input = Input(shape=(*IMAGE_SIZE, IMAGE_CHANNELS))

    x = Conv2D(16, 4, 2, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, 4, 2, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, 4, 2, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, 4, 2, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(_input, x)
    model.summary()
    return model

if __name__ == '__main__':
    data_loader = CIFAR10DataLoader()
    classifier_model = define_classifier()
    classifier_model.compile(optimizer=Adam(lr=1e-3, beta_1=.9, clipvalue=10., clipnorm=10.),
                             loss='categorical_crossentropy', metrics=['accuracy'])

    loss_list     = []
    accuracy_list = []
    epochs = 30
    batch_size = 2048
    for epoch in range(epochs):
        data_generator, steps = data_loader.batch_generator(batch_size=batch_size)
        for step in range(steps):
            x, y = next(data_generator)
            [loss, accuracy] = classifier_model.train_on_batch(x, y)
            print(f'epoch={epoch}, step={step}, loss: {loss}, accuracy: {accuracy}')
            loss_list.append(loss)
            accuracy_list.append(accuracy)

    plt.figure((16, 16))
    plt.plot(loss_list)
    plt.plot(accuracy_list)
    plt.legend(['loss', 'accuracy'])
    plt.show()