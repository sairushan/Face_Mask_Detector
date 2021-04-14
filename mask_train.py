import numpy as np
import cv2
from tensorflow import keras
from google.colab.patches import cv2_imshow
from tensorflow.keras.layers import Dense,Activation,Flatten,Conv2D,MaxPool2D,Dropout
from keras.callbacks import ModelCheckpoint

def augment(img):
  M=cv2.getRotationMatrix2D((25,25),np.random.randint(-10,11),1)
  img=cv2.warpAffine(img,M,(50,50))
  return img


path="/content/drive/MyDrive/FaceMask/Mask_train.npy"
dataset=np.load(path,allow_pickle=True)
print(dataset.shape)

train_inputs=[]
train_targets=[]
for img,target in dataset[:1800]:
  train_inputs.append(augment(img))
  train_targets.append(target)
train_inputs=np.array(train_inputs)
train_targets=np.array(train_targets)
normalised_train_inputs=train_inputs/255

print(train_inputs.shape)
print(train_targets.shape)

model = keras.Sequential() 
model.add(Conv2D(32,(3,3),padding="same",input_shape=normalised_train_inputs.shape[1:])) #50x50x3
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2))) 
model.add(Conv2D(64,(3,3),padding="same")) 
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2))) 
model.add(Conv2D(128,(3,3),padding="same")) 
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2))) 
model.add(Conv2D(256,(3,3),padding="same",))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2))) 
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=0.001),metrics=['accuracy'])

filepath="/content/drive/MyDrive/FaceMask/Models/best.hdf5"

checkpoint=ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
callbacks_list=[checkpoint]
model.fit(normalised_train_inputs,train_targets,validation_split=0.05,batch_size=32,epochs=30,callbacks=callbacks_list,verbose=1)
