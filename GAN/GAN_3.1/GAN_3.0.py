import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten, Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import warnings , os
warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import umap
import umap.plot

img_rows = 256
img_cols = 256
channels = 3

# input image dimension
img_shape = (img_cols, img_rows, channels)

# size of noise vector to be used as generator input
z_dim = 128

# Setting Hyperparameter
iterations = 100000
sample_interval = 200
batch_size = 16

image_grid_rows=4
image_grid_columns=4

times = 250000



def build_generator(z_dim):
    model = Sequential()
    
    model.add(Dense(z_dim, input_dim=z_dim))
    model.add(Reshape((1, 1, z_dim)))


    model.add(Conv2DTranspose(1024, kernel_size=4, strides=4, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2DTranspose(1024, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2DTranspose(512, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(Activation('tanh'))

    return model



def build_discriminator(img_shape):
    model = Sequential()

    model.add(Conv2D(64, kernel_size=4, strides=4, input_shape=img_shape, padding='same', use_bias=False))
    model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(256, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(512, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(1024, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(1024, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(1, kernel_size=4, strides=2, padding='same', use_bias=False))
    
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model



# Creating and Compiling discriminator Models
discriminator = build_discriminator(img_shape)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator.load_weights('discriminator_size256_model.h5')

# Create a generator model
generator = build_generator(z_dim)
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator.load_weights('generator_size256_model.h5')


def discriminator_loss(real, generated):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(real), real)

    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5

def generator_loss(generated):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(generated), generated)


losses = []
accuracies = []
iteration_checkpoints = []

def train(iterations, batch_size, sample_interval):
    # Load MNIST dataset
    size=256

    x_train=np.loadtxt("anime_size"+str(size)+".npy")
    #print(file_np.shape)

    x_train=np.array_split(x_train,int(x_train.shape[0]/size))
    x_train=np.array(x_train,dtype=np.dtype(np.int32))
    #print(x_train.shape)

    x_train=np.array_split(x_train,int(x_train.shape[0]/3))
    x_train=np.array(x_train,dtype=np.dtype(np.int32))
    x_train=x_train.transpose(0,2,3,1)
    print(x_train.shape)


    
    # [0, 255] scales black and white pixel values between [-1, 1]
    X_train = x_train / 127.5 - 1.0

    # Real Image Label: All 1
    real = np.ones((batch_size, 1))

    # Fake Image Labels: All 0
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):
        with tf.GradientTape(persistent=True) as tape:

            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            z = np.random.normal(0, 1, (batch_size, z_dim))
            gen_imgs = generator(z, training=True)

            d_loss_real = discriminator(imgs, training=True)
            d_loss_fake = discriminator(gen_imgs, training=True)

            disc_loss = discriminator_loss(d_loss_real, d_loss_fake)



            gen_loss = generator_loss(d_loss_fake)


        discriminator_gradients = tape.gradient(disc_loss,discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,discriminator.trainable_variables))

        generator_gradients = tape.gradient(gen_loss,generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(generator_gradients,generator.trainable_variables))


        if (iteration + 1) % 10000 == 0:
            generator.save("generator_size"+str(img_rows)+"_"+str(iteration + 1 + times)+"_model.h5")
            discriminator.save("discriminator_size"+str(img_rows)+"_"+str(iteration + 1 + times)+"_model.h5")

##        if iteration < 1000 and (iteration + 1) % 20 == 0:
##            print(iteration + 1)
##
##            # Generated image sample output
##            sample_images(iteration + 1)
            
        if (iteration + 1) % sample_interval == 0:
            print(iteration + 1)

            # Generated image sample output
            sample_images(iteration + 1)



def sample_images(epochs):
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, (z_dim)))
    gen_imgs = generator.predict(z)  
    gen_imgs = 0.5 * gen_imgs + 0.5

    _, ax = plt.subplots(image_grid_rows,image_grid_columns)
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            gen_imgs[i*image_grid_rows+j] = cv2.cvtColor(gen_imgs[i*image_grid_rows+j], cv2.COLOR_BGR2RGB)
            ax[i,j].imshow(gen_imgs[i*image_grid_rows+j])
            ax[i,j].axis('off')
    plt.savefig("model_size"+str(img_rows)+"_save/animefaces_"+str(batch_size)+"/animefaces_"+str(epochs + times)+".png")
    #plt.show()



# DCGAN training for a specified number of iterations
train(iterations, batch_size, sample_interval)

generator.save('generator_size64_model.h5')
discriminator.save('discriminator_size64_model.h5')
