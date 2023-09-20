import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import numpy as np
import os

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)

AUTOTUNE = tf.data.experimental.AUTOTUNE
    
print(tf.__version__)



##datagen = ImageDataGenerator(
##    featurewise_center=False,  #在數據集上將輸入均值設置為0
##    samplewise_center=False,  #將每個樣本均值設置為0
##    featurewise_std_normalization=False,  #按特徵將輸入除以數據集的標準值
##    samplewise_std_normalization=False,  #將每個樣本輸入除以其標準
##    zca_whitening=False,  #ZCA白化(PCA) 強調?
##    zca_epsilon=1e-06,  #ZCA白化的epsilon值 設1e-6
##    rotation_range=0,  #[0,指定角度]範圍内角度隨機旋轉(範圍0到180) 
##    width_shift_range=0.1,  #隨機平移 圖片寬的尺寸乘以參數
##    height_shift_range=0.1,  #隨機上下移 圖片長的尺寸乘以參數
##    shear_range=0.,  #錯位變換 就歪?
##    zoom_range=0.,  #大小改變 0~1放大 1~?縮小 也可以放兩個參數
##    channel_shift_range=0.,  #整體改變顏色
##    fill_mode='nearest',  #填補圖像資料擴增時造成的像素缺失
##    cval=0.,  #用於邊界外的值 fill_mode="constant"適用
##    horizontal_flip=True,  #隨機水平翻轉
##    vertical_flip=False,  #隨機上下翻轉
##    rescale=None, #對圖片的每個像素乘上這個放縮因子 (1/255 0~1間)
##    preprocessing_function=None,  #應用於每個輸入的函數 ?
##    data_format=None,  #"channels_first"或"channels_last" ?
##    validation_split=0.0)  #用於驗證的圖像分數 0~1 ?

datagen = ImageDataGenerator(
    rotation_range=45,  #[0,指定角度]範圍内角度隨機旋轉(範圍0到180) 
    width_shift_range=0.1,  #隨機平移 圖片寬的尺寸乘以參數
    height_shift_range=0.1,  #隨機上下移 圖片長的尺寸乘以參數
    zoom_range=0.8,  #大小改變 0~1放大 1~?縮小 也可以放兩個參數
    fill_mode='nearest',  #填補圖像資料擴增時造成的像素缺失
    cval=0.,  #用於邊界外的值 fill_mode="constant"適用
    horizontal_flip=True)  #隨機水平翻轉


size=256

#selfie_train=np.loadtxt("images_size256.npy")
selfie_train=np.loadtxt("selfie2anime_data/images_selfie.npy")
#print(file_np.shape)

selfie_train=np.array_split(selfie_train,int(selfie_train.shape[0]/size))
selfie_train=np.array(selfie_train,dtype=np.dtype(np.float32))
#print(selfie_train.shape)

selfie_train=np.array_split(selfie_train,int(selfie_train.shape[0]/3))
selfie_train=np.array(selfie_train,dtype=np.dtype(np.float32))
selfie_train=selfie_train.transpose(0,2,3,1)
#print(selfie_train.shape)

#selfie_train = selfie_train / 127.5 - 1.0

selfie_dataset=[]
selfie_gan=datagen.flow(selfie_train,batch_size=1)
for i_gan in range(selfie_gan.__len__()):
    selfie_dataset.append(selfie_gan.next()[0])
selfie_dataset=np.array(selfie_dataset,dtype=np.dtype(np.float32))
selfie_dataset = selfie_dataset / 127.5 - 1.0

##print(selfie_dataset[0])
##plt.imshow(selfie_dataset[0])
##plt.show()

selfie_dataset = tf.data.Dataset.from_tensor_slices(selfie_dataset).batch(1)

###datagen1 = datagen
##xoox=datagen.flow(selfie_train)
###datagen1.fit(anime_train)
##print(xoox.__len__())
##print(xoox.next())
##print(type(xoox.next()))
##print(selfie_dataset)
###print(datagen.shape)



#anime_train=np.loadtxt("images_size256.npy")
anime_train=np.loadtxt("selfie2anime_data/images_anime.npy")
#print(file_np.shape)

anime_train=np.array_split(anime_train,int(anime_train.shape[0]/size))
anime_train=np.array(anime_train,dtype=np.dtype(np.float32))
#print(selfie_train.shape)

anime_train=np.array_split(anime_train,int(anime_train.shape[0]/3))
anime_train=np.array(anime_train,dtype=np.dtype(np.float32))
anime_train=anime_train.transpose(0,2,3,1)
#print(selfie_train.shape)

#anime_train = anime_train / 127.5 - 1.0

anime_dataset=[]
anime_gan=datagen.flow(anime_train,batch_size=1)
for i_gan in range(anime_gan.__len__()):
    anime_dataset.append(anime_gan.next()[0])
anime_dataset=np.array(anime_dataset,dtype=np.dtype(np.float32))
anime_dataset = anime_dataset / 127.5 - 1.0

##print(anime_dataset[0])
##plt.imshow(anime_dataset[0])
##plt.show()

anime_dataset = tf.data.Dataset.from_tensor_slices(anime_dataset).batch(1)



IMAGE_SIZE = [256, 256]
OUTPUT_CHANNELS = 3

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
    last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
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
with strategy.scope():
    monet_generator = Generator() # transforms photos to Monet-esque paintings
    photo_generator = Generator() # transforms Monet paintings to be more like photos

    monet_discriminator = Discriminator() # differentiates real Monet paintings and generated Monet paintings
    photo_discriminator = Discriminator() # differentiates real photos and generated photos

    monet_generator.load_weights('anime_generator_model.h5')
    photo_generator.load_weights('selfie_generator_model.h5')
    monet_discriminator.load_weights('anime_discriminator_model.h5')
    photo_discriminator.load_weights('selfie_discriminator_model.h5')

    #to_monet = monet_generator(example_photo)



class CycleGan(keras.Model):
    def __init__(
        self,
        monet_generator,
        photo_generator,
        monet_discriminator,
        photo_discriminator,
        lambda_cycle=10,
    ):
        super(CycleGan, self).__init__()
        self.m_gen = monet_generator
        self.p_gen = photo_generator
        self.m_disc = monet_discriminator
        self.p_disc = photo_discriminator
        self.lambda_cycle = lambda_cycle
        
    def compile(
        self,
        m_gen_optimizer,
        p_gen_optimizer,
        m_disc_optimizer,
        p_disc_optimizer,
        gen_loss_fn,
        disc_loss_fn,
        cycle_loss_fn,
        identity_loss_fn
    ):
        super(CycleGan, self).compile()
        self.m_gen_optimizer = m_gen_optimizer
        self.p_gen_optimizer = p_gen_optimizer
        self.m_disc_optimizer = m_disc_optimizer
        self.p_disc_optimizer = p_disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn
        
    def train_step(self, batch_data):
        real_monet, real_photo = batch_data
        
        with tf.GradientTape(persistent=True) as tape:
            # photo to monet back to photo
            fake_monet = self.m_gen(real_photo, training=True)
            cycled_photo = self.p_gen(fake_monet, training=True)

            # monet to photo back to monet
            fake_photo = self.p_gen(real_monet, training=True)
            cycled_monet = self.m_gen(fake_photo, training=True)

            # generating itself
            same_monet = self.m_gen(real_monet, training=True)
            same_photo = self.p_gen(real_photo, training=True)

            # discriminator used to check, inputing real images
            disc_real_monet = self.m_disc(real_monet, training=True)
            disc_real_photo = self.p_disc(real_photo, training=True)

            # discriminator used to check, inputing fake images
            disc_fake_monet = self.m_disc(fake_monet, training=True)
            disc_fake_photo = self.p_disc(fake_photo, training=True)

            # evaluates generator loss
            monet_gen_loss = self.gen_loss_fn(disc_fake_monet)
            photo_gen_loss = self.gen_loss_fn(disc_fake_photo)

            # evaluates total cycle consistency loss
            total_cycle_loss = self.cycle_loss_fn(real_monet, cycled_monet, self.lambda_cycle) + self.cycle_loss_fn(real_photo, cycled_photo, self.lambda_cycle)

            # evaluates total generator loss
            total_monet_gen_loss = monet_gen_loss + total_cycle_loss + self.identity_loss_fn(real_monet, same_monet, self.lambda_cycle)
            total_photo_gen_loss = photo_gen_loss + total_cycle_loss + self.identity_loss_fn(real_photo, same_photo, self.lambda_cycle)

            # evaluates discriminator loss
            monet_disc_loss = self.disc_loss_fn(disc_real_monet, disc_fake_monet)
            photo_disc_loss = self.disc_loss_fn(disc_real_photo, disc_fake_photo)

        # Calculate the gradients for generator and discriminator
        monet_generator_gradients = tape.gradient(total_monet_gen_loss,
                                                  self.m_gen.trainable_variables)
        photo_generator_gradients = tape.gradient(total_photo_gen_loss,
                                                  self.p_gen.trainable_variables)

        monet_discriminator_gradients = tape.gradient(monet_disc_loss,
                                                      self.m_disc.trainable_variables)
        photo_discriminator_gradients = tape.gradient(photo_disc_loss,
                                                      self.p_disc.trainable_variables)

        # Apply the gradients to the optimizer
        self.m_gen_optimizer.apply_gradients(zip(monet_generator_gradients,
                                                 self.m_gen.trainable_variables))

        self.p_gen_optimizer.apply_gradients(zip(photo_generator_gradients,
                                                 self.p_gen.trainable_variables))

        self.m_disc_optimizer.apply_gradients(zip(monet_discriminator_gradients,
                                                  self.m_disc.trainable_variables))

        self.p_disc_optimizer.apply_gradients(zip(photo_discriminator_gradients,
                                                  self.p_disc.trainable_variables))
        
        return {
            "monet_gen_loss": total_monet_gen_loss,
            "photo_gen_loss": total_photo_gen_loss,
            "monet_disc_loss": monet_disc_loss,
            "photo_disc_loss": photo_disc_loss
        }

with strategy.scope():
    def discriminator_loss(real, generated):
        real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(real), real)

        generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(generated), generated)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss * 0.5

with strategy.scope():
    def generator_loss(generated):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(generated), generated)

with strategy.scope():
    def calc_cycle_loss(real_image, cycled_image, LAMBDA):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

        return LAMBDA * loss1

with strategy.scope():
    def identity_loss(real_image, same_image, LAMBDA):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return LAMBDA * 0.5 * loss

with strategy.scope():
    monet_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    monet_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

with strategy.scope():
    cycle_gan_model = CycleGan(
        monet_generator, photo_generator, monet_discriminator, photo_discriminator
    )

    cycle_gan_model.compile(
        m_gen_optimizer = monet_generator_optimizer,
        p_gen_optimizer = photo_generator_optimizer,
        m_disc_optimizer = monet_discriminator_optimizer,
        p_disc_optimizer = photo_discriminator_optimizer,
        gen_loss_fn = generator_loss,
        disc_loss_fn = discriminator_loss,
        cycle_loss_fn = calc_cycle_loss,
        identity_loss_fn = identity_loss
    )

    

times=80
data_path="selfie2anime_data/model_save"
class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (epoch+1)%5 == 0:
            os.makedirs(os.path.join(data_path, "epochs_"+str(epoch+1+times)))
            monet_generator.save(data_path+"/epochs_"+str(epoch+1+times)+"/anime_generator_model.h5")
            photo_generator.save(data_path+"/epochs_"+str(epoch+1+times)+"/selfie_generator_model.h5")
            monet_discriminator.save(data_path+"/epochs_"+str(epoch+1+times)+"/anime_discriminator_model.h5")
            photo_discriminator.save(data_path+"/epochs_"+str(epoch+1+times)+"/selfie_discriminator_model.h5")
saver = CustomSaver()

cycle_gan_model.fit(
    tf.data.Dataset.zip((anime_dataset, selfie_dataset)),
    callbacks=[saver],
    epochs=100
)

monet_generator.save('anime_generator_model.h5')
photo_generator.save('selfie_generator_model.h5')
monet_discriminator.save('anime_discriminator_model.h5')
photo_discriminator.save('selfie_discriminator_model.h5')
