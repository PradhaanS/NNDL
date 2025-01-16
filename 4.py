import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10, cifar100

def load_data(dataset, num_classes):
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test

def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),

        Dense(num_classes, activation='softmax')
    ])

    return model

def train_and_evaluate(dataset, num_classes, epochs=10, batch_size=64):
    x_train, y_train, x_test, y_test = load_data(dataset, num_classes)

    model = build_cnn(x_train.shape[1:], num_classes)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    datagen.fit(x_train)

    model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        validation_data=(x_test, y_test),
        epochs=epochs
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    print(f"Test Accuracy: {acc:.2f}")

    return model

train_and_evaluate(cifar10, num_classes=10)

train_and_evaluate(cifar100, num_classes=100)
