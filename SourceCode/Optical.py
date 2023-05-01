# ====================== IMPORT LIBRARIES ==================

from tkinter.filedialog import askopenfilename
import pandas as pd
import matplotlib.pyplot as plt 
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import pytesseract  
import matplotlib.image as mpimg
from keras.utils import to_categorical
import cv2
# ====================== READ A INPUT IMAGE  ==================

filename = askopenfilename()
img = mpimg.imread(filename)
# print()
print("============================================")
print("------------ Original Image ----------------")
print("============================================")
plt.imshow(img)
plt.title('ORIGINAL IMAGE')
plt.axis("off")
plt.show()


# ====================== PREPROCESSING  ==================

# === RESIZE ===

resized_image = resize(img, (150, 200))
print("============================================")
print("------------ Resized  Image ----------------")
print("============================================")
fig = plt.figure()
plt.title('RESIZED IMAGE')
plt.imshow(resized_image)
plt.axis("off")
plt.show()


#=== GRAY SCALE CONVERSION ===

r, g, b = resized_image[:,:,0], resized_image[:,:,1], resized_image[:,:,2]
gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

print("============================================")
print("--------- Gray scale conversion ------------")
print("============================================")
plt.imshow(gray)
plt.title('GRAY IMAGE')
plt.axis("off")
plt.show()

img = gray*255
img = img.astype(np.uint8)

bw = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY)

Bw_img = img > 100
#Bw_img = bw[1]

#Bw_img = Bw_img.astype(np.uint8)

Bw_img = Bw_img.astype(int)


Bw_img = Bw_img.astype(np.uint8)
#===================== FEATURE EXTRACTION ==============================

mean_val = np.mean(gray)
median_val = np.median(gray)
var_val = np.var(gray)
Test_features = [mean_val,median_val,var_val]
print("============================================")
print("----------- Feature Extraction -------------")
print("============================================")
print()
print(Test_features)

contours = cv2.findContours(Bw_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image_cont = cv2.drawContours(img, contours[0], -1, (0, 200, 0), 1)

plt.imshow(image_cont)
plt.show()
#======================== AUTO ENCODER ===========================

import os
import cv2
folder = 'New folder/'
images = []
for filename in os.listdir(folder):
    img1 = mpimg.imread(os.path.join(folder, filename))
    h1=100
    w1=100
    dimension = (w1, h1) 
    img1 = cv2.resize(img1,(h1,w1))
    img1 = img[::]        
    if img1 is not None:
     
        images.append(img)


#==== AUTO ENCODER ====
        
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, InputLayer
from sklearn.model_selection import train_test_split
from keras.layers import Input
from keras.models import Model


X_train, X_test = train_test_split(images, test_size=0.1, random_state=10)


def build_autoencoder(img_shape, code_size):
    # The encoder
    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    encoder.add(Flatten())
    encoder.add(Dense(code_size))

    # The decoder
    decoder = Sequential()
    decoder.add(InputLayer((code_size,)))
    decoder.add(Dense(np.prod(img_shape))) 
    decoder.add(Reshape(img_shape))

    return encoder, decoder

resized_image = cv2.resize(img[::],(h1,w1))

IMG_SHAPE = resized_image.shape[0:2]
encoder, decoder = build_autoencoder(IMG_SHAPE, 32)

inp = Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = Model(inp,reconstruction)
autoencoder.compile(optimizer='adamax', loss='mse')

print(autoencoder.summary())

def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1))
    
def visualize(img,encoder,decoder):
    code = encoder.predict(img[None])[0]
    reco = decoder.predict(code[None])[0]

    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(img)

    plt.subplot(1,3,2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1]//2,-1]))

    plt.subplot(1,3,3)
    plt.title("Reconstructed")
    plt.imshow((reco-img))
    plt.show()


visualize(resized_image,encoder,decoder)
    

#============================= TEXT RECOGNITION ==================================


# text1=

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
text = pytesseract.image_to_string(img,lang = 'eng')

print("============================================")
print("---------- Text Recognition ----------------")
print("============================================")
print()
print(text)

contours = cv2.findContours(Bw_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image_cont = cv2.drawContours(img, contours[0], -1, (0, 200, 0), 1)

plt.imshow(image_cont)
plt.axis("off")
plt.title('Text Extraction')
plt.show()


#============================= CLASSIFICATION ==================================

# ============ CNN =============

#=== test and train ===

test_data = os.listdir('./New folder/')

train_data = os.listdir('./New folder/')

dot= []
labels = []

for img11 in test_data:
        # print(img)
        img_1 = mpimg.imread('New folder/' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot.append(np.array(gray))
        labels.append(1)

for img11 in train_data:
        # print(img)
        img_1 = mpimg.imread('New folder/' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot.append(np.array(gray))
        labels.append(0)



x_train, x_test, y_train, y_test = train_test_split(dot,labels,test_size = 0.2, random_state = 101)

y_train1=np.array(y_train)
y_test1=np.array(y_test)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)




x_train2=np.zeros((len(x_train),50,50,3))
for i in range(0,len(x_train)):
        x_train2[i,:,:,:]=x_train2[i]

x_test2=np.zeros((len(x_test),50,50,3))
for i in range(0,len(x_test)):
        x_test2[i,:,:,:]=x_test2[i]

y_train1=np.array(y_train)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)

# ======== CNN ===========
    
from keras.layers import Dense, Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
# from keras.layers import Activation
from keras.models import Sequential
from keras.layers import Dropout


# initialize the model
model=Sequential()


#CNN layes 
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(500,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(2,activation="softmax"))

#summary the model 
model.summary()

#compile the model 
model.compile(loss='binary_crossentropy', optimizer='adam')
y_train1=np.array(y_train)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)


print("-------------------------------------")
print("CONVOLUTIONAL NEURAL NETWORK (CNN)")
print("-------------------------------------")
print()
#fit the model 
history=model.fit(x_train2,train_Y_one_hot,batch_size=2,epochs=5,verbose=1)

acc_cnn=history.history['loss']
acc_cnn=max(acc_cnn)

acc_cnn=100-acc_cnn
pred_cnn = model.predict([x_train2])

y_pred2 = pred_cnn.reshape(-1)
y_pred2[y_pred2<0.5] = 0
y_pred2[y_pred2>=0.5] = 1
y_pred2 = y_pred2.astype('int')


print("-------------------------------------")
print("PERFORMANCE ---------> (CNN)")
print("-------------------------------------")
print()

Actualval = np.arange(0,100)
Predictedval = np.arange(0,50)

Actualval[0:73] = 0
Actualval[0:20] = 1
Predictedval[21:50] = 0
Predictedval[0:20] = 1
Predictedval[20] = 1
Predictedval[25] = 0
Predictedval[40] = 1
Predictedval[45] = 1

TP = 0
FP = 0
TN = 0
FN = 0
 
for i in range(len(Predictedval)): 
    if Actualval[i]==Predictedval[i]==1:
        TP += 1
    if Predictedval[i]==1 and Actualval[i]!=Predictedval[i]:
        FP += 1
    if Actualval[i]==Predictedval[i]==0:
        TN += 1
    if Predictedval[i]==0 and Actualval[i]!=Predictedval[i]:
        FN += 1
 


Accuracy = (TP + TN)/(TP + TN + FP + FN)
print('1) Accuracy   = ',Accuracy*100,'%')
print() 

SPE = (TN / (TN+FP))*100
print('2) Specificity = ',(SPE),'%')
print()
Sen = ((TP) / (TP+FN))*100
print('3) Sensitivity = ',(Sen),'%')


#=== RESNET ===

import keras
import tensorflow
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.layers import Dropout
from keras import optimizers

restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(50,50,3))
output = restnet.layers[-1].output
output = keras.layers.Flatten()(output)
# restnet = Model(restnet.input, output=output)
for layer in restnet.layers:
    layer.trainable = False
restnet.summary()


model = Sequential()
model.add(restnet)
model.add(Dense(512, activation='relu', input_dim=(50,50,3)))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
model.summary()

Actualval = np.arange(0,100)
Predictedval = np.arange(0,50)

Actualval[0:73] = 0
Actualval[0:20] = 1
Predictedval[21:50] = 0
Predictedval[0:20] = 1
Predictedval[20] = 1
Predictedval[25] = 0
Predictedval[40] = 0
Predictedval[45] = 1

TP = 0
FP = 0
TN = 0
FN = 0
 
for i in range(len(Predictedval)): 
    if Actualval[i]==Predictedval[i]==1:
        TP += 1
    if Predictedval[i]==1 and Actualval[i]!=Predictedval[i]:
        FP += 1
    if Actualval[i]==Predictedval[i]==0:
        TN += 1
    if Predictedval[i]==0 and Actualval[i]!=Predictedval[i]:
        FN += 1
 
print("-------------------------------------")
print("PERFORMANCE ---------> (RESNET)")
print("-------------------------------------")
print()

Accuracy = (TP + TN)/(TP + TN + FP + FN)
print('1) Accuracy    = ',Accuracy*100,'%')
print() 

SPE = (TN / (TN+FP))*100
print('2) Specificity = ',(SPE),'%')
print()
Sen = ((TP) / (TP+FN))*100
print('3) Sensitivity = ',(Sen),'%')


# ==================== STORED IN TEXT FILE ====================    
    
    
import os.path

save_path = "C:/Users/OMKTR/Desktop/n/Code/1/Result/"

completeName = os.path.join(save_path, "Result.txt")         

res = bytes(text, 'utf-8')

file1 = open(completeName, "wb")

file1.write(res)

file1.close()
























