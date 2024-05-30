# print('Setting Up')
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# from utils import importDataInfo
# from utils import balanceData
# from utils import loadData
# from utils import augmentImage
# from utils import createModel
# from utils import batchGen

# ### step 1
# path = 'data'
# data = importDataInfo(path)

# ### step 2
# data=balanceData(data, display=False)

# ### step 3
# imagesPath, steering=loadData(path,data)
# # print(imagesPath[0], steering[0])

# ### step 4
# xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steering, test_size=0.2, random_state=5)
# print('Total Training Images:', len(xTrain))
# print('Total Validation Images:', len(xVal))

# ### step 5
# # imgRe, st = augmentImage(imagesPath[0], steering[0])
# # plt.imshow(imgRe)
# # plt.show()

# #### step 6

# #### step 7

# ### setp 8
# model =createModel()
# model.summary()

# ### step 9
# model.fit(batchGen(xTrain,yTrain,10,1),steps_per_epoch=20,epoches=2,
#         validation_data=batchGen(xVal,yVal,10,0), validation_steps=20)


print('Setting Up')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import importDataInfo
from utils import balanceData
from utils import loadData
from utils import augmentImage
from utils import createModel
from utils import batchGen

### step 1
path = 'data'
data = importDataInfo(path)

### step 2
data = balanceData(data, display=False)

### step 3
imagesPath, steering = loadData(path, data)
# print(imagesPath[0], steering[0])

### step 4
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steering, test_size=0.2, random_state=5)
print('Total Training Images:', len(xTrain))
print('Total Validation Images:', len(xVal))

### step 5
# imgRe, st = augmentImage(imagesPath[0], steering[0])
# plt.imshow(imgRe)
# plt.show()

### step 6
# Your code for step 6 goes here (if any)

### step 7
# Your code for step 7 goes here (if any)

### step 8
model = createModel()
model.summary()

### step 9
history = model.fit(batchGen(xTrain, yTrain, 100, 1), steps_per_epoch=300, epochs=10,
        validation_data=batchGen(xVal, yVal, 100, 0), validation_steps=200)

### sep 10
model.save('model.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'validation'])
plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()