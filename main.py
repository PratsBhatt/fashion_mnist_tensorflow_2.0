import tensorflow as tf
from model import create_model
import numpy as np

# constants
NORMALIZE_FACTOR = 255
BATCH_SIZE = 16

# loading the dataset, using tensorflow dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test ) = fashion_mnist.load_data()

# Normalizing the data, to make the values between 0-1
x_train, x_test = x_train / NORMALIZE_FACTOR , x_test / NORMALIZE_FACTOR

# As to use Conv2D layer, we need the data in the N x W X H X C (4 dimensions)
# We will have to extend the dimensions, adding a color channel
x_train, x_test = np.expand_dims(x_train, -1), np.expand_dims(x_test, -1)

# Get the number of classes
num_classes = len(set(y_train))

# loss function
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2, mode='min')
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs', update_freq='batch',
                                             write_graph=False, profile_batch=2)
save_checkpoints = tf.keras.callbacks.ModelCheckpoint('./checkpints', monitor='loss', save_best_only=True,
                                                      mode='min', save_freq='epoch')

# create model
model = create_model(x_train[0].shape, num_classes)

# compile model
model.compile('adam', loss=loss, metrics=['accuracy'])

# Print model summary
model.summary()

# Prepare the data, add image augmentation for improved results
# Let's create generator
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, rotation_range=20,
                                                                  width_shift_range=0.1, height_shift_range=0.2)

train_generator = image_generator.flow(x_train, y_train, BATCH_SIZE)

r = model.fit(train_generator, epochs=50,
          validation_data=(x_test, y_test), callbacks=[early_stopping, tensorboard, save_checkpoints])

print(' Training is completed')
