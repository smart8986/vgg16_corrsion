
from keras.models import load_model
import cv2
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
resize = 224
 
def jundge(predicted):
    predicted = np.argmax(predicted)
    if predicted == 0:
        print('cat')
    else:
        print('dog')
load_model = load_model("./vgg16dogcat.h5")  # 读取模型
img = cv2.imread('cat.png')  # 读入灰度图
 
img = cv2.resize(img, (resize, resize))
img = img.reshape(-1,resize,resize,3)
img = img.astype('float32')
predicted = load_model.predict(img)  # 输出预测结果
print(predicted)
