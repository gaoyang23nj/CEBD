import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

def show_test_data():
    show_list = [0, 1, 2]
    plt.figure(figsize=(height * 1 * 1.2, width * 3 * 1.6))
    plt.subplot(1, 3, 1)
    tmp_index = show_list[0]
    plt.imshow(x_valid[tmp_index], interpolation='nearest')
    plt.title(test_predict_class[tmp_index])
    plt.subplot(1, 3, 2)
    tmp_index = show_list[1]
    plt.imshow(x_valid[tmp_index], interpolation='nearest')
    plt.title(test_predict_class[tmp_index])
    plt.subplot(1, 3, 3)
    tmp_index = show_list[2]
    plt.imshow(x_valid[tmp_index], interpolation='nearest')
    plt.title(test_predict_class[tmp_index])
    plt.show()

def show_history():
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["loss", "val_loss", "accuracy", "val_accuracy"], loc="upper left")
    plt.show()

x = tf.random.uniform([1, 1])
tmp = x.device.endswith('GPU:0')
print('On GPU:{}'.format(tmp))

# (x_train_pre, y_train_pre), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
(x_train, y_train), (x_valid, y_valid) = tf.keras.datasets.cifar10.load_data()
width = 32
height = 32
channel = 3
num_classes = 10

class_name = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

x_train_scaled = x_train / 255.
x_valid_scaled = x_valid / 255.

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu',
                                 input_shape=[width, height, channel]))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=2))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=2))
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

# resnet50_fine_tune.add(tf.keras.applications.ResNet50(include_top = False,
#                                                       pooling = 'avg',
#                                                       weights = 'imagenet'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
model.summary()


history = model.fit(x_train_scaled, y_train, epochs=100, validation_data=(x_valid_scaled, y_valid), verbose=2)

evaluate_res = model.evaluate(x_valid_scaled, y_valid, verbose=2)
predict_res = model.predict(x_valid_scaled)
test_predict_class_indices = np.argmax(predict_res, axis=1)
test_predict_class = [class_name[index] for index in test_predict_class_indices]

