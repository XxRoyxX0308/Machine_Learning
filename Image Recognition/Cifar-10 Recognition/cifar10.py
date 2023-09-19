from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

import numpy as np
import cv2

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from keras.utils import np_utils

from sklearn.metrics import confusion_matrix, classification_report
import itertools

#==========================設定==========================
batch_size = 32  #優化一次次數
num_classes = 10  #測資種類
epochs = 100  #全部重跑幾次
data_augmentation = True  #數據增強 多
#========================================================

#==========================數據==========================
#數據輸入 訓練 測試 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
cv2.imwrite("a.jpg", x_train[4])
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
#========================================================

'''
#==========================圖表==========================
fig, axs = plt.subplots(1,2,figsize=(15,5))  #圖表初始化
sns.countplot(x=y_train.ravel(), ax=axs[0])  #算有幾張訓練圖
axs[0].set_title('Distribution of training data')  #標題
axs[0].set_xlabel('Classes')  #x軸文字
sns.countplot(x=y_test.ravel(), ax=axs[1])  #算有幾張測試圖
axs[1].set_title('Distribution of Testing data')  #標題
axs[1].set_xlabel('Classes')  #x軸文字
plt.show()
#========================================================
'''

#========================數據處理========================
x_train = x_train.astype('float32')  #修改資料類別
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train, num_classes)  #轉數列(二進制?)
y_test = np_utils.to_categorical(y_test, num_classes)
#========================================================

#==========================主體==========================
model = Sequential()  #定義神經網絡

# CONV => RELU => CONV => RELU => POOL => DROPOUT
model.add(Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# CONV => RELU => CONV => RELU => POOL => DROPOUT
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# FLATTERN => DENSE => RELU => DROPOUT
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# a softmax classifier
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary() #輸出神經結構
#========================================================

#==========================訓練==========================
#啟動RMSprop優化器
#opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

#opt = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)  #!!
#opt = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0) #!
#opt = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0) #學很快 效果X
opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#opt = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
#opt = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

#loss 優化器編譯
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
#========================================================

#========================圖片增強========================
history = None  #記錄訓練過程
if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(x_train, y_train,  #執行訓練
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),  #驗證圖片
              shuffle=True)  #打亂測資順序 懶惰模型
else:
    print('Using real-time data augmentation.')
    #數據增強 TURE
    datagen = ImageDataGenerator(
        featurewise_center=False,  #在數據集上將輸入均值設置為0
        samplewise_center=False,  #將每個樣本均值設置為0
        featurewise_std_normalization=False,  #按特徵將輸入除以數據集的標準值
        samplewise_std_normalization=False,  #將每個樣本輸入除以其標準
        zca_whitening=False,  #ZCA白化(PCA) 強調?
        zca_epsilon=1e-06,  #ZCA白化的epsilon值 設1e-6
        rotation_range=0,  #[0,指定角度]範圍内角度隨機旋轉(範圍0到180) 
        width_shift_range=0.1,  #隨機平移 圖片寬的尺寸乘以參數
        height_shift_range=0.1,  #隨機上下移 圖片長的尺寸乘以參數
        shear_range=0.,  #錯位變換 就歪?
        zoom_range=0.,  #大小改變 0~1放大 1~?縮小 也可以放兩個參數
        channel_shift_range=0.,  #整體改變顏色
        fill_mode='nearest',  #填補圖像資料擴增時造成的像素缺失
        cval=0.,  #用於邊界外的值 fill_mode="constant"適用
        horizontal_flip=True,  #隨機水平翻轉
        vertical_flip=False,  #隨機上下翻轉
        rescale=None, #對圖片的每個像素乘上這個放縮因子 (1/255 0~1間)
        preprocessing_function=None,  #應用於每個輸入的函數 ?
        data_format=None,  #"channels_first"或"channels_last" ?
        validation_split=0.0)  #用於驗證的圖像分數 0~1 ?

    #計算特徵標準化所需的數量
    #如果應用ZCA白化 則為標準差、均值和主成分
    datagen.fit(x_train)

    #fit簡單數據 fit_generator較複雜數據 train_on_batch自訂
    history = model.fit_generator(datagen.flow(x_train, y_train,  #datagen.flow 生成圖片
                                    batch_size=batch_size),
                                    epochs=epochs,
                                    validation_data=(x_test, y_test),
                                    workers=4)  #GPU平行運算
#========================================================

#========================訓練圖表========================
def plotmodelhistory(history): 
    fig, axs = plt.subplots(1,2,figsize=(15,5)) 

    axs[0].plot(history.history['accuracy'])  #資料輸入
    axs[0].plot(history.history['val_accuracy']) 
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy') 
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['train', 'validate'], loc='upper left')  #說明線條顏色

    axs[1].plot(history.history['loss']) 
    axs[1].plot(history.history['val_loss']) 
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss') 
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'validate'], loc='upper left')
    plt.show()

print(history.history.keys())  #輸出所有數據

plotmodelhistory(history)
#========================================================

#========================測試模型========================
scores = model.evaluate(x=x_test, y=y_test, verbose=1)  #評分
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

pred = model.predict(x_test)  #預測
#========================================================

#========================錯誤測資========================
def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    """
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    ax.set_xlabel('Predicted Label') 
    ax.set_ylabel('True Label')
    
    return im, cbar

def annotate_heatmap(im, data=None, fmt="d", threshold=None):
    """
    A function to annotate a heatmap.
    """
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = im.axes.text(j, i, format(data[i, j], fmt), horizontalalignment="center",
                                 color="white" if data[i, j] > thresh else "black")
            texts.append(text)

    return texts

labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(pred, axis=1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_test, axis=1)
# Errors are difference between predicted labels and true labels
errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = pred[errors]
Y_true_errors = Y_true[errors]
X_test_errors = x_test[errors]

cm = confusion_matrix(Y_true, Y_pred_classes) 
thresh = cm.max() / 2.

fig, ax = plt.subplots(figsize=(12,12))
im, cbar = heatmap(cm, labels, labels, ax=ax,
                   cmap=plt.cm.Blues, cbarlabel="count of predictions")
texts = annotate_heatmap(im, data=cm, threshold=thresh)

fig.tight_layout()
plt.show()

print(classification_report(Y_true, Y_pred_classes))
#========================================================

#========================檢查預測========================
R = 5
C = 5
fig, axes = plt.subplots(R, C, figsize=(12,12))
axes = axes.ravel()

for i in np.arange(0, R*C):
    axes[i].imshow(x_test[i])
    axes[i].set_title("True: %s \nPredict: %s" % (labels[Y_true[i]], labels[Y_pred_classes[i]]))
    axes[i].axis('off')
    plt.subplots_adjust(wspace=1)
#========================================================

#======================檢查錯誤預測======================
R = 3
C = 5
fig, axes = plt.subplots(R, C, figsize=(12,8))
axes = axes.ravel()

misclassified_idx = np.where(Y_pred_classes != Y_true)[0]
for i in np.arange(0, R*C):
    axes[i].imshow(x_test[misclassified_idx[i]])
    axes[i].set_title("True: %s \nPredicted: %s" % (labels[Y_true[misclassified_idx[i]]], 
                                                  labels[Y_pred_classes[misclassified_idx[i]]]))
    axes[i].axis('off')
    plt.subplots_adjust(wspace=1)
#========================================================

#======================檢查重要錯誤======================
def display_errors(errors_index, img_errors, pred_errors, obs_errors):
    """ This function shows 10 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 5
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True, figsize=(12,6))
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((32,32,3)))
            ax[row,col].set_title("Predicted:{}\nTrue:{}".
                                  format(labels[pred_errors[error]],labels[obs_errors[error]]))
            n += 1
            ax[row,col].axis('off')
            plt.subplots_adjust(wspace=1)

# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 10 errors 
most_important_errors = sorted_dela_errors[-10:]

# Show the top 10 errors
display_errors(most_important_errors, X_test_errors, Y_pred_classes_errors, Y_true_errors)
#========================================================

#==========================測試==========================
def show_test(number):
    fig = plt.figure(figsize = (3,3))
    test_image = np.expand_dims(x_test[number], axis=0)
    test_result = model.predict_classes(test_image)
    plt.imshow(x_test[number])
    dict_key = test_result[0]
    plt.title("Predicted: {} \nTrue Label: {}".format(labels[dict_key],
                                                      labels[Y_true[number]]))
show_test(20)
#========================================================

#========================處存檔案========================
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
#========================================================
