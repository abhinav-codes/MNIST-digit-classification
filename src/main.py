import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

# as_supervised loads the Dataset as a tuple(key:value)
# mnist_dataset is 'dict'
mnist_dataset, mnist_information = tfds.load(name='mnist', as_supervised = 'TRUE', with_info = 'TRUE')
mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

# Assigns valdation sample as 10% of 60000
num_validation_samples = 0.1*mnist_information.splits['train'].num_examples
num_test_samples = mnist_information.splits['test'].num_examples

# Typecasting the Data into integer
num_validation_samples = tf.cast(num_validation_samples, tf.int64)
num_test_samples = tf.cast(num_test_samples, tf.int64)

# Defining function of scaling
def scale(image,label):
    image = tf.cast(image, tf.float32)
    image = image/255
    #print(image,label)
    return image,label

# scaling MNIST using the map()
scaled_train_and_validation_data =mnist_dataset['train'].map(scale)
scaled_test_data = mnist_dataset['test'].map(scale)

# To get a good Dataset we assign a Buffer Size to shuffle the data. 
# BUFFER_SIZE = 1 No shuffle
# BUFFER_SIZe >= Sample Uniform

BUFFER_SIZE = 10000
shuffled_scaled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE) 

# Taking out the data take / skip
validation_data = shuffled_scaled_train_and_validation_data.take(num_validation_samples)
train_data = shuffled_scaled_train_and_validation_data.skip(num_validation_samples)

# Create Batches
BATCH_SIZE = 100
train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(num_validation_samples)

test_data = scaled_test_data.batch(num_test_samples)
validation_inputs, validation_targets = next(iter(validation_data))

# Define variables for the model
INPUT_SIZE = 784 # 28 x 28 x 1
OUTPUT_SIZE = 10
HIDDEN_LAYER_SIZE = 1000

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),# input layer
    tf.keras.layers.Dense(HIDDEN_LAYER_SIZE, activation='relu'),
    tf.keras.layers.Dense(HIDDEN_LAYER_SIZE, activation='relu'),    
    tf.keras.layers.Dense(HIDDEN_LAYER_SIZE, activation='relu'),
    tf.keras.layers.Dense(HIDDEN_LAYER_SIZE, activation='relu'),
    tf.keras.layers.Dense(OUTPUT_SIZE, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
              
NUM_EPOCHS = 7

model.fit(train_data, epochs=NUM_EPOCHS, validation_data=(validation_inputs, validation_targets), verbose =2, validation_steps=10)
          
# Test the Model
test_loss, test_accuracy = model.evaluate(test_data)
