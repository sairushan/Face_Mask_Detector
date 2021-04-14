import numpy as np
import cv2
import os


parent="FaceImages/"
folders=os.listdir(parent)

dataset=[]

classes=["Mask","NoMask"]
for folder in folders:
    full_path=os.path.join(parent,folder)
    target=classes.index(folder)
    imgs=os.listdir(full_path)
    for img in imgs:
        try:
            img_path=os.path.join(full_path,img)
            img=cv2.imread(img_path)
            resized_img=cv2.resize(img,(50,50))
            dataset.append([resized_img,target])
            #cv2.imshow("img",resized_img)
            #cv2.waitKey(1)
        except:
            print("error")
dataset=np.array(dataset)
print(dataset.shape)
np.save("Mask_train.npy",dataset)
