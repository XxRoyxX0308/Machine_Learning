import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import cv2
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

img_rows = 64
img_cols = 64
channels = 3

# input image dimension
img_shape = (img_rows, img_cols, channels)

# size of noise vector to be used as generator input
z_dim = 100



def build_generator(z_dim):
    model = Sequential()
    model.add(Dense(256 * 16 * 16, input_dim=z_dim))
    model.add(Reshape((16, 16, 256)))
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2DTranspose(3, kernel_size=3, strides=2, padding='same'))
    model.add(Activation('tanh'))

    return model



def build_discriminator(img_shape):
    model = Sequential()
    model.add(
        Conv2D(32,
               kernel_size=3,
               strides=2,
               input_shape=img_shape,
               padding='same'))

    model.add(LeakyReLU(alpha=0.01))

    model.add(
        Conv2D(64,
               kernel_size=3,
               strides=2,
               padding='same'))

    model.add(LeakyReLU(alpha=0.01))

    model.add(
        Conv2D(128,
               kernel_size=3,
               strides=2,
               padding='same'))

    model.add(LeakyReLU(alpha=0.01))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model



def build_gan(generator, discriminator):
    model = Sequential()

    # Genearator -> Discriminator
    model.add(generator)
    model.add(discriminator)

    return model



# Creating and Compiling discriminator Models
discriminator = build_discriminator(img_shape)
#discriminator.load_weights('discriminator_size64_model.h5')

discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])
# Preserve the parameters of the discriminator while training the generator
discriminator.trainable = False

# Create a generator model
generator = build_generator(z_dim)
#generator.load_weights('generator_size64_model.h5')

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



    size=64

    x_train=np.loadtxt("images_size"+str(size)+".npy")
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
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)

        # Training 
        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Training generator
        # ---------------------

        # Create fake image batches
        z = np.random.normal(0, 1, (batch_size, 100))
        
        #if iteration == 0:
            #print("\nRandom noise input image")
            #sample_images(x_train[idx],y_train[idx],generator)            

        g_loss = gan.train_on_batch(z, real)

        if (iteration + 1) % 10000 == 0:
            generator.save("generator_size64_"+str(iteration + 1)+"_model.h5")
            discriminator.save("discriminator_size64_"+str(iteration + 1)+"_model.h5")

        if iteration < 1000 and (iteration + 1) % 20 == 0:
            # Save loss and accuracy to plot graphs after training
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)

            print("%d [D loss: %f, accuracy: %.2f%%] [G loss: %f]" %
                  (iteration + 1, d_loss, 100.0 * accuracy, g_loss))

            # Generated image sample output
            sample_images(iteration + 1)
            
        elif (iteration + 1) % sample_interval == 0:
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
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))
    gen_imgs = generator.predict(z)  
    gen_imgs = 0.5 * gen_imgs + 0.5

    _, ax = plt.subplots(image_grid_rows,image_grid_columns)
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            gen_imgs[i*image_grid_rows+j] = cv2.cvtColor(gen_imgs[i*image_grid_rows+j], cv2.COLOR_BGR2RGB)
            ax[i,j].imshow(gen_imgs[i*image_grid_rows+j])
            ax[i,j].axis('off')
    plt.savefig("model_size64_save/animefaces_16/animefaces_"+str(epochs)+".png")
    #plt.show()



# Setting Hyperparameter
iterations = 10000
batch_size = 16
sample_interval = 200

# DCGAN training for a specified number of iterations
train(iterations, batch_size, sample_interval)

generator.save('generator_size64_model.h5')
discriminator.save('discriminator_size64_model.h5')

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
