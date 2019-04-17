from __future__ import division
import numpy as np
import glob
from scipy.spatial import distance

def std_normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x
    """
    meanX = np.mean(x)
    stdX = np.std(x)
    x = (x-meanX)/stdX
    return x
def normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def prepare_data_mask(mask):
    offset = int(mask.shape[0] / 2)
    print('Training Data Preparation Start...')
    #x_t = np.zeros((mask.shape[0],30,30),dtype='float')
    #x_t[0:x_t.shape[0],1:29,1:29] = mask
    #x_t = std_normalize(x_t)
    x_t = mask[0:offset, :,:]

    '''
    print('Reading Mask Instances...')
    x_t=np.load('/media/siddique/Data/CLASP2018/CAE_train_mask/10A/mask_exp10acam9_270.npy',encoding='bytes')
    #x_t = np.load('/media/siddique/Data/CLASP2018/CAE_train_mask/10A/mask_exp10acam11_270.npy', encoding='bytes')
    '''

    #cam9: 13000, 13790, cam11: 10000, 11209
    #x_train = x_t[0:12000,:,:].astype('float32')# / 255. #normalize pixel values between 0 and 1 [coarse mask has already in 0-1 scale]
    #x_test = x_t[12000:13790,:,:].astype('float32')# / 255.
    x_train = np.reshape(x_t, (len(x_t),mask.shape[1] ,mask.shape[2], 1))  # adapt this if using `channels_first` image data format
    #x_test = np.reshape(x_test, (len(x_test), img_y,img_x, 1))  # adapt this if using `channels_first` image data format
    print('Training Data Preparation Done...')
    return x_train

def prepare_data_box(x_t):
    print('Training Data Preparation Start...')
    offset = int(x_t.shape[0]/2)

    x_0 = x_t[0:offset, 2]/ 1920 #/ 1920#np.max(x_t[:, 2])
    y_0 = x_t[0:offset, 3]/ 1080 #/ 1080#np.max(x_t[:, 3])
    w = x_t[0:offset, 4]/ 1920 #/ 1920#np.max(x_t[:, 4])
    h = x_t[0:offset, 5]/ 1080# / 1080#np.max(x_t[:, 5])
    Cx =(x_t[0:offset, 2] + x_t[0:offset, 4] / 2)/ 1920 #/ 1920# np.max((x_t[:, 2] + x_t[:, 4] / .2))
    Cy = (x_t[0:offset, 3] + x_t[0:offset, 5] / 2)/ 1080 #/ 1080# np.max((x_t[:, 2] + x_t[:, 4] / .2))
    area = (x_t[0:offset, 4] * x_t[0:offset, 5])/ (1920*1080)#np.max((x_t[:, 4] * x_t[:, 5]))
    diag = np.sqrt(x_t[0:offset, 4]**2 + x_t[0:offset, 5]**2)/np.sqrt(1920**2+1080**2)
    score = x_t[:, 6]
    # prepare dim = 8:[Cx,Cy,x,y,w,h,wh,class]
    #x_t = np.array([Cx, Cy, w, h])
    x_t = np.array([Cx, Cy, w, h,x_0,y_0,area,diag])#, for mot x_0,y_0 instead of w,h
    x_t = np.transpose(x_t)
    #x_t = normalize(x_t)

    print('Training Data Preparation Done...')
    return x_t

path = '/media/siddique/Data/CLASP2018/train_data_all/mask_*' #frame149
files = glob.glob(path)
files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
masks_all = []
boxs_all = []
for name in files:
    print(name)
    #x_train_box = prepare_data_box()
    x_train = np.load(name)
    x_train_mask=prepare_data_mask(x_train)
    masks_all.append(x_train_mask)
masks_all = np.array(masks_all)
box_list = [b for b in masks_all]
masks_all = np.concatenate(box_list)
np.save('/media/siddique/Data/CLASP2018/train_data_all/train_mask_mot',masks_all,allow_pickle=True,fix_imports=True)
#np.save('/media/siddique/Data/CLASP2018/train_data_all/train_mask_mot',masks_all,allow_pickle=True,fix_imports=True)
print(masks_all.shape)