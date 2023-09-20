import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import layers
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
img_shape = (img_rows, img_cols, channels)

# size of noise vector to be used as generator input
z_dim = 100

##############################################################################

def build_generator(z_dim):
    model = Sequential()
    model.add(Dense(256 * 4 * 4, input_dim=z_dim))
    model.add(Reshape((4, 4, 256)))
    
    model.add(Conv2DTranspose(512, kernel_size=4, strides=2, kernel_initializer=tf.random_normal_initializer(0., 0.02), padding='same'))
    model.add(BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Conv2DTranspose(512, kernel_size=4, strides=2, kernel_initializer=tf.random_normal_initializer(0., 0.02), padding='same'))
    model.add(BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Conv2DTranspose(512, kernel_size=4, strides=2, kernel_initializer=tf.random_normal_initializer(0., 0.02), padding='same'))
    model.add(BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Conv2DTranspose(256, kernel_size=4, strides=2, kernel_initializer=tf.random_normal_initializer(0., 0.02), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, kernel_initializer=tf.random_normal_initializer(0., 0.02), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, kernel_initializer=tf.random_normal_initializer(0., 0.02), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    model.add(Conv2DTranspose(16, kernel_size=2, strides=1, kernel_initializer=tf.random_normal_initializer(0., 0.02), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Conv2DTranspose(3, kernel_size=2, strides=1, kernel_initializer=tf.random_normal_initializer(0., 0.02), padding='same'))
    model.add(Activation('tanh'))

    return model



def build_discriminator(img_shape):
    model = Sequential()
    model.add(
        Conv2D(64,
               kernel_size=4,
               strides=2,
               input_shape=img_shape,
               padding='same'))

    #model.add(LeakyReLU(alpha=0.01))

    model.add(
        Conv2D(128,
               kernel_size=4,
               strides=2,
               padding='same'))

    #model.add(LeakyReLU(alpha=0.01))

    model.add(
        Conv2D(256,
               kernel_size=4,
               strides=2,
               padding='same'))

    #model.add(layers.ZeroPadding2D())

    model.add(
        Conv2D(512,
               kernel_size=4,
               strides=2,
               padding='same'))

    model.add(tfa.layers.InstanceNormalization(gamma_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
    model.add(LeakyReLU(alpha=0.01))

    model.add(
        Conv2D(1,4,
               strides=2,
               kernel_initializer=tf.random_normal_initializer(0., 0.02),
               padding='same'))
    #model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))



##    model = Sequential()
##    model.add(
##        Conv2D(32,
##               kernel_size=3,
##               strides=2,
##               input_shape=img_shape,
##               padding='same'))
##
##    model.add(LeakyReLU(alpha=0.01))
##
##    model.add(
##        Conv2D(64,
##               kernel_size=3,
##               strides=2,
##               padding='same'))
##
##    model.add(LeakyReLU(alpha=0.01))
##
##    model.add(
##        Conv2D(128,
##               kernel_size=3,
##               strides=2,
##               padding='same'))
##
##    model.add(LeakyReLU(alpha=0.01))
##
##    model.add(Flatten())
##    model.add(Dense(1, activation='sigmoid'))

    
    return model

##############################################################################

def downsample(filters, size, apply_instancenorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_instancenorm:
        result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    result.add(layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())

    return result

def Generator():
    inputs = layers.Input(shape=[256,256,3])

    # bs = batch size
    down_stack = [
        downsample(64, 4, apply_instancenorm=False), # (bs, 128, 128, 64)
        downsample(128, 4), # (bs, 64, 64, 128)
        downsample(256, 4), # (bs, 32, 32, 256)
        downsample(512, 4), # (bs, 16, 16, 512)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4), # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4), # (bs, 16, 16, 1024)
        upsample(256, 4), # (bs, 32, 32, 512)
        upsample(128, 4), # (bs, 64, 64, 256)
        upsample(64, 4), # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(channels, 4,
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh') # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)

    return keras.Model(inputs=inputs, outputs=x)

def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = layers.Input(shape=[256, 256, 3], name='input_image')

    x = inp

    down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

    zero_pad1 = layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = layers.Conv2D(512, 4, strides=1,
                         kernel_initializer=initializer,
                         use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    norm1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)

    leaky_relu = layers.LeakyReLU()(norm1)

    zero_pad2 = layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    last = layers.Conv2D(1, 4, strides=1,
                         kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=inp, outputs=last)

##############################################################################

def build_gan(generator, discriminator):
    model = Sequential()

    # Genearator -> Discriminator
    model.add(generator)
    model.add(discriminator)

    return model



# Creating and Compiling discriminator Models
#discriminator = build_discriminator(img_shape)
discriminator = Discriminator()

#discriminator.load_weights('discriminator_size256_40000_model.h5')

discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])
# Preserve the parameters of the discriminator while training the generator
discriminator.trainable = False

# Create a generator model
#generator = build_generator(z_dim)
generator = Generator()

#generator.load_weights('generator_size256_40000_model.h5')

# Create and compile a GAN model with frozen discriminators to train the generator
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

losses = []
accuracies = []
iteration_checkpoints = []

def train(iterations, batch_size, sample_interval):
    # Load MNIST dataset
##    (x_train, y_train), (_, _) = mnist.load_data()
##    print(x_train)
##    print(type(x_train))
##    print(x_train[0])
##    print(type(x_train[0]))

    

##    size=64
##    x_train=[]
##
##    image_paths = [f for f in os.listdir(r"images") if os.path.isfile(os.path.join(r"images", f))]
##    for image_path in image_paths:
##        x_train.append(cv2.resize(cv2.imread("images/"+image_path),dsize=(size,size)))
##        print(image_path)
##    x_train=np.array(x_train)



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
    #X_train = np.expand_dims(X_train, axis=3)

    # Real Image Label: All 1
    real = np.ones((batch_size, 1))

    # Fake Image Labels: All 0
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):
        # -------------------------
        #  Discriminant training
        # -------------------------
        
        # Get random batches from real images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        # Create fake image batches
        #z = np.random.normal(0, 1, (batch_size, 100))
        z = np.random.normal(0, 1, (batch_size, 256 ,256 ,3))
        gen_imgs = generator.predict(z)

        # Training 
        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Training generator
        # ---------------------

        # Create fake image batches
        #z = np.random.normal(0, 1, (batch_size, 100))
        z = np.random.normal(0, 1, (batch_size, 256 ,256 ,3))
        
        #if iteration == 0:
            #print("\nRandom noise input image")
            #sample_images(x_train[idx],y_train[idx],generator)            

        g_loss = gan.train_on_batch(z, real)

        if (iteration + 1) % 10000 == 0:
            generator.save("generator_size256_"+str(iteration + 1)+"_model.h5")
            discriminator.save("discriminator_size256_"+str(iteration + 1)+"_model.h5")

##        if iteration < 1000 and (iteration + 1) % 20 == 0:
##            # Save loss and accuracy to plot graphs after training
##            losses.append((d_loss, g_loss))
##            accuracies.append(100.0 * accuracy)
##            iteration_checkpoints.append(iteration + 1)
##
##            print("%d [D loss: %f, accuracy: %.2f%%] [G loss: %f]" %
##                  (iteration + 1, d_loss, 100.0 * accuracy, g_loss))
##
##            # Generated image sample output
##            sample_images(iteration + 1)
            
        if (iteration + 1) % sample_interval == 0:
            # Save loss and accuracy to plot graphs after training
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)

            print("%d [D loss: %f, accuracy: %.2f%%] [G loss: %f]" %
                  (iteration + 1, d_loss, 100.0 * accuracy, g_loss))

            # Generated image sample output
            sample_images(iteration + 1)




image_grid_rows=4
image_grid_columns=4

def sample_images(epochs):
    #z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, 256, 256, 3))
    gen_imgs = generator.predict(z)  
    gen_imgs = 0.5 * gen_imgs + 0.5

    _, ax = plt.subplots(image_grid_rows,image_grid_columns)
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            gen_imgs[i*image_grid_rows+j] = cv2.cvtColor(gen_imgs[i*image_grid_rows+j], cv2.COLOR_BGR2RGB)
            ax[i,j].imshow(gen_imgs[i*image_grid_rows+j])
            ax[i,j].axis('off')
    plt.savefig("model_change_size256_save/batch_size_1/animefaces_"+str(epochs)+".png")
    #plt.show()



# Setting Hyperparameter
iterations = 100000
batch_size = 1
sample_interval = 200

# DCGAN training for a specified number of iterations
train(iterations, batch_size, sample_interval)

generator.save('generator_size256_model.h5')
discriminator.save('discriminator_size256_model.h5')

##accuracies = np.array(accuracies)
##
### Discriminator Accuracy Graph
##plt.figure(figsize=(15, 5))
##plt.plot(iteration_checkpoints, accuracies, label="Discriminator accuracy",linewidth=2)
##
##plt.xticks(iteration_checkpoints, rotation=90)
##plt.yticks(range(0, 100, 10))
##
##plt.title("Discriminator Accuracy",fontsize = 20)
##plt.xlabel("Iteration")
##plt.ylabel("Accuracy (%)")
##plt.legend()
