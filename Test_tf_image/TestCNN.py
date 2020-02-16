import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import sklearn

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
width = 32
height = 32
channel = 3
num_classes = np.max(y_train) - np.min(y_train) + 1

class_name = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
x_valid, y_valid = x_train[-5000:], y_train[-5000:]
x_train, y_train = x_train[:-5000], y_train[:-5000]
x_train_scaled = x_train / 255.
x_valid_scaled = x_valid / 255.
x_test_scaled = x_test / 255.

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu',
                                 input_shape=[width, height, channel]))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2))

model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

# resnet50_fine_tune.add(tf.keras.applications.ResNet50(include_top = False,
#                                                       pooling = 'avg',
#                                                       weights = 'imagenet'))

model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(x_train_scaled, y_train, epochs=10, validation_data=(x_valid_scaled, y_valid))

evaluate_res = model.evaluate(x_test_scaled, y_test, verbose=2)
predict_res = model.predict(x_test_scaled)
test_predict_class_indices = np.argmax(predict_res, axis=1)
test_predict_class = [class_name[index] for index in test_predict_class_indices]

show_list = [0, 1, 2]
plt.figure(figsize=(height*1*1.2, width*3*1.6))
plt.subplot(0, 0, 1)
tmp_index = show_list[0]
plt.imshow(x_test[tmp_index])
plt.title(test_predict_class[tmp_index])

plt.subplot(0, 1, 2)
tmp_index = show_list[1]
plt.imshow(x_test[tmp_index])
plt.title(test_predict_class[tmp_index])

plt.subplot(0, 2, 3)
tmp_index = show_list[2]
plt.imshow(x_test[tmp_index])
plt.title(test_predict_class[tmp_index])

plt.show()