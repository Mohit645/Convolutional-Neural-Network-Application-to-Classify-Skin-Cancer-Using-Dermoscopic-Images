# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 15:42:03 2019

@author: Mohit
"""

import os
from glob import glob
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

print(os.listdir)
file_repository="D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer"
#Changing the working directory.
os.chdir(file_repository)
#Getting the path of all the images in the working directory.
image_paths = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(file_repository, '*', '*.jpg'))}
#We created a dictionary of the skin cancer lesions.
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
#Reading the meta data we have.
skincancerdf = pd.read_csv(os.path.join(file_repository, 'HAM10000_metadata.csv'))
skincancerdf.head()
len(skincancerdf)
#Making a new column in dataframe to accomodate the images path to perform preprocessing required.
skincancerdf['image_path'] = skincancerdf['image_id'].map(image_paths.get)
skincancerdf['type'] = skincancerdf['dx'].map(lesion_type_dict.get)
 
#Transforming the type to the categorical values
skincancerdf['type_id'] = pd.Categorical(skincancerdf['type']).codes

#Getting the count per class to use for data augmentation stage.
nv_count=0
mel_count=0
bkl_count=0
bcc_count=0
akiec_count=0
vasc_count=0
df_count=0
for i in range(len(skincancerdf)):
    if(skincancerdf['dx'][i]=='bkl'):
        bkl_count=bkl_count+1
        
    elif(skincancerdf['dx'][i]=='nv'):
        nv_count=nv_count+1
        
    elif(skincancerdf['dx'][i]=='mel'):
        mel_count=mel_count+1
        
    elif(skincancerdf['dx'][i]=='bcc'):
        bcc_count=bcc_count+1
        
    elif(skincancerdf['dx'][i]=='akiec'):
        akiec_count=akiec_count+1
        
    elif(skincancerdf['dx'][i]=='df'):
        df_count=df_count+1
        
    elif(skincancerdf['dx'][i]=='vasc'):
        vasc_count=vasc_count+1
        
print(nv_count,mel_count,bkl_count,bcc_count,akiec_count,vasc_count,df_count)
#Validating total number of images should be equal to 10,015.
print(nv_count+mel_count+bkl_count+bcc_count+akiec_count+vasc_count+df_count)

#Number of times image augmentation has to be done.
#Since NV has highest
diff_mel=nv_count-mel_count
diff_bkl=nv_count-bkl_count
diff_bcc=nv_count-bcc_count
diff_akiec=nv_count-akiec_count
diff_vasc=nv_count-vasc_count
diff_df=nv_count-df_count

print(diff_akiec+diff_bcc+diff_bkl+diff_df+diff_mel+diff_vasc)

#Calculating the number of iterations required for image augmentation 
mel_itr = round(diff_mel/mel_count)
bkl_itr = round(diff_bkl/bkl_count)
bcc_itr = round(diff_bcc/bcc_count)
akiec_itr = round(diff_akiec/akiec_count)
vasc_itr = round(diff_vasc/vasc_count)
df_itr = round(diff_df/df_count)

#First we will perform morphological operation to remove noise and hair.
cancerNames = list()
cancerNames = ["MEL", "NV", "AKIEC", "BCC", "BKL", "DF", "VASC"]
import glob
import shutil
import os
for i in cancerName:
    os.mkdir('D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\'+i)
    
skincancerdf["image_path"]
for index, row in skincancerdf.iterrows():   
#Splitting the images as per the class names.
    skincancerdf["image_path"]
    for index, row in skincancerdf.iterrows():
        if(row["dx"] == 'vasc'):        
            src_dir='D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\HAM10000\\'+row["image_id"]+'.jpg'
            dst_dir = "D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\VASC"
            shutil.copy(src_dir,dst_dir) 
            
        elif(row["dx"]=='nv'):
            src_dir='D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\HAM10000\\'+row["image_id"]+'.jpg'
            dst_dir = "D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\NV"
            shutil.copy(src_dir,dst_dir) 
        
        
        elif(row["dx"]=='mel'):
            src_dir='D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\HAM10000\\'+row["image_id"]+'.jpg'
            dst_dir = "D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\MEL"
            shutil.copy(src_dir,dst_dir) 
        
        elif(row["dx"]=='bcc'):
            src_dir='D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\HAM10000\\'+row["image_id"]+'.jpg'
            dst_dir = "D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\BCC"
            shutil.copy(src_dir,dst_dir) 
        
        elif(row["dx"]=='akiec'):
            src_dir='D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\HAM10000\\'+row["image_id"]+'.jpg'
            dst_dir = "D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\AKIEC"
            shutil.copy(src_dir,dst_dir) 
        
        elif(row["dx"]=='df'):
            src_dir='D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\HAM10000\\'+row["image_id"]+'.jpg'
            dst_dir = "D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\DF"
            shutil.copy(src_dir,dst_dir) 
        
        
        elif(row["dx"]=='bkl'):
            src_dir='D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\HAM10000\\'+row["image_id"]+'.jpg'
            dst_dir = "D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\BKL"
            shutil.copy(src_dir,dst_dir) 
    


for i in cancerNames:
    path = "D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\"+i
    images = [f for f in os.listdir(path) if os.path.splitext(f)[-1] == '.jpg']
	for image in images:
        img = cv2.imread('D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\'+i+'\\'+image)
        kernel = np.ones((5,5), np.uint8)
        img_erosion = cv2.erode(img, kernel, iterations=1)
        img_dilation = cv2.dilate(img, kernel, iterations=2)
        cv2.imwrite('D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\'+i+'\\'+image, img_dilation)
	
#Images will get updated in the same folder and overwrite the original images with the same image names.



#Image Augmentation Code 
#We did not keep 'NV' Class in cancer list below as we dont want to augment it.
cancerNames=list()
cancerNames=['MEL','BCC','BKL','DF','VASC','AKIEC']
for i in cancerNames:
    path = "D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\"+i
    images = [f for f in os.listdir(path) if os.path.splitext(f)[-1] == '.jpg']   
    
    if(i=="MEL"):
        itr=mel_itr
    elif(i=="VASC"):
        itr=vasc_itr
    elif(i=="BCC"):
        itr=bcc_itr
    elif(i=="BKL"):
        itr=bkl_itr
    elif(i=="DF"):
        itr=df_itr
    elif(i=="AKIEC"):
        itr=akiec_itr       


    for image in images:
        img = load_img("D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\"+i+"\\"+image) 
        data = img_to_array(img)
        samples = expand_dims(data, 0)
        datagen = ImageDataGenerator(rotation_range=90)
        it = datagen.flow(samples, batch_size=32, save_to_dir='D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\'+i ,save_prefix='AUG_ISIC_GEN_'+image, save_format='jpg')
        for j in range(itr):
            batch = it.next()

#Validating the count of images.
cancerNames = list()
cancerNames = ["MEL", "NV", "AKIEC", "BCC", "BKL", "DF", "VASC"]

for i in cancerNames:
  path = 'D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\'+i
  num_files = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])
  print(i)
  print(num_files)


cancerNames = list()
cancerNames = ["MEL", "NV", "AKIEC", "BCC", "BKL", "DF", "VASC"]
#Making Test and Train folder and splitting data class wise.
os.mkdir('D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\Dataset\\Train')
os.mkdir('D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\Dataset\\Test')
#Below code will automatically create class folder and copy images.
import shutil
from sklearn.model_selection import train_test_split
import os
for i in cancerNames:
    
    path = 'D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\'+i
    
    images = [f for f in os.listdir(path) if os.path.splitext(f)[-1] == '.jpg']
    train,test = train_test_split(images, test_size = 1/4, random_state = 0)
    
    dst_dir = "D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\Dataset\\Train\\"+i
    os.mkdir(dst_dir)
    for j in train:
        src_dir="D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\"+i+"\\"+j
        shutil.copy(src_dir,dst_dir)
     
    dst_dir = "D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\Dataset\\Test\\"+i 
    os.mkdir(dst_dir)
    for k in test:
        src_dir="D:\\NCI\\Sem2\\1.ADM\\skin cancer\\skin cancer\\"+i+"\\"+k
        shutil.copy(src_dir,dst_dir)
    del(images)

#Building The CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import os
#Initialising CNN

CNNmodel = Sequential()

#Step1 Convolution
#We defined the input image size as 32,32 to reduce the operation cost.
CNNmodel.add(Convolution2D(32,3,3, input_shape=(32,32,3), activation='relu'))

#Step2 Pooling
CNNmodel.add(MaxPool2D(pool_size=(2,2)))

#Adding dropout layer
#To refrain our model to overfit.
CNNmodel.add(Dropout(0.25))

#added second conv layer so that we can extract features more efficiently.
CNNmodel.add(Convolution2D(32,3,3, input_shape=(32,32,3), activation='relu'))
CNNmodel.add(MaxPool2D(pool_size=(2,2)))
CNNmodel.add(Dropout(0.25))
#Step3 Flattening
CNNmodel.add(Flatten())

#Step 4 Full Connection
CNNmodel.add(Dense(output_dim= 128, activation= 'relu'))
#Since we are dealing in multiple classes we set the ac function as softmax.
CNNmodel.add(Dense(output_dim= 7, activation= 'softmax'))

#Step 5
CNNmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#We placed the images in SSD from HDD to reduce the operation cost. 
file_repository="C:\\Users\\Mohit\\OneDrive\\Documents\\Segmented"
os.chdir(file_repository)

#Step6 Fitting images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

from keras.callbacks import ReduceLROnPlateau
annealer  = ReduceLROnPlateau(monitor='val_acc')


training_set = train_datagen.flow_from_directory('Train', target_size=(32,32), batch_size=32, class_mode='categorical')

test_set = test_datagen.flow_from_directory('Test', target_size=(32,32), batch_size=32, class_mode='categorical')
#We added the loss function for the model. Since we have multiple classes so we used categorical crossentropy function.
CNNmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Our model was generated here and this also gave us per epoch output.
CNNmodel.fit_generator(training_set,
                       steps_per_epoch=35116,
                       epochs=5,
                       validation_data=test_set,
                       validation_steps=11709,
                       callbacks=[annealer])
