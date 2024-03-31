#Gabor Features Independent

#import libraries and modules
from spectral import imshow, view_cube
import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
from spectral import ndvi
import scipy.ndimage as ndimage
from skimage import filters
import skimage
import copy
import numpy.ma as ma
import pandas as pd
from sklearn.decomposition import PCA
from scipy import ndimage, misc
import matplotlib.pyplot as plt
from skimage import data
from skimage import filters
from skimage import exposure
from spectral import kmeans
import seaborn as sns
from spectral import principal_components
from sklearn.decomposition import PCA
from skimage import viewer
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
import math
from sklearn import preprocessing
import scipy
from cv2 import filter2D as _do_convolve

#define functions
def calculate_mean_masked_spectra(reflArray,cop,cop_threshold,ineq='>'):

    mean_masked_refl = np.zeros(reflArray.shape[2])

    for i in np.arange(reflArray.shape[2]):
        refl_band = reflArray[:,:,i]

        if ineq == '>':
            cop_mask = ma.masked_where((cop<=cop_threshold) | (np.isnan(cop)),cop)
        elif ineq == '<':
            cop_mask = ma.masked_where((cop>=cop_threshold) | (np.isnan(cop)),cop)   
        else:
            print('ERROR: Invalid inequality. Enter < or >')

        masked_refl = ma.MaskedArray(refl_band,mask=cop_mask.mask)
        mean_masked_refl[i] = ma.mean(masked_refl)

    return mean_masked_refl

def snv(input_data):
    data_snv = np.zeros_like(input_data)
    for i in range(data_snv.shape[0]):
        data_snv[i] = (input_data[i] - np.mean(input_data)) / np.std(input_data)
    return (data_snv)


def masked_spectra(reflArray, mask):
    
    M = np.zeros(reflArray.shape)

    for i in np.arange(reflArray.shape[2]):
        refl_band = reflArray[:,:,i]
        M_band =M[:,:,i]
        M_band[mask] = refl_band[mask]
        

    return M


#START PROCESSING

#read .raw file
data_ref = envi.open(r"C:\Users\FlaviaMihaela\Desktop\PLNT_BRSNN_CHRGR_INOCL_21dai_06.hdr", r"C:\Users\FlaviaMihaela\Desktop\PLNT_BRSNN_CHRGR_INOCL_21dai_06.raw")
data= np.array(data_ref.load())


#apply laplacian-gaussian blur then otsu thresholding
copy_data=data[:,:,25]
#plt.hist(copy_data)
#plt.show()
copy_data_log=ndimage.gaussian_laplace(copy_data, sigma=11)
#plt.imshow(copy_data,cmap='gray', interpolation='nearest')
thres = np.absolute(copy_data_log).mean() * 0.75
output = np.zeros(copy_data_log.shape)
w = output.shape[1]
h = output.shape[0]
for y in range(1, h - 1):
    for x in range(1, w - 1):
        patch = copy_data_log[y-1:y+2, x-1:x+2]
        p = copy_data_log[y, x]
        maxP = patch.max()
        minP = patch.min()
        if (p > 0):
            zeroCross = True if minP < 0 else False
        else:
           zeroCross = True if maxP > 0 else False
        if ((maxP - minP) > thres) and zeroCross:
            output[y, x] = 1
#plt.rcParams["axes.grid"] = False
#plt.imshow(output, cmap='gray')
#plt.show()
output=output.astype('bool')
sel = np.zeros_like(copy_data)
sel[output] = copy_data[output]
#plt.hist(sel)
#plt.show()


#Multiple thresholding using otsu's method
#m_otsu_t= filters.threshold_multiotsu(sel)
#regions = np.digitize(copy_data, bins=m_otsu_t)
#plt.imshow(regions, cmap='Accent')
#plt.show()
#print(m_otsu_t)


#Obtaining the mask
spare_data=copy_data
spare_data[spare_data<52.29492] = np.nan
mask_log=(copy_data>52.29492)
#viewer = skimage.viewer.ImageViewer(mask_log)
#viewer.show()
sel2 = np.zeros_like(copy_data)
sel2[mask_log] = copy_data[mask_log]
#plt.imshow(sel2, cmap='gray', interpolation='nearest')


#Obtain the segmented 3D image by applying the mask on all bands
m_data=masked_spectra(data,mask_log)
#v1= imshow(m_data, (12,9,3), stretch=(0.03, 0.99), figsize= (5,5))
#plt.show()


#Generate gabor filter bank
num=1
df_r=pd.DataFrame()
df_i=pd.DataFrame()
frequency=0.2
for i in np.arange(m_data.shape[2]):
    test = m_data[:,:,i]
    for theta in np.arange(0, np.pi, np.pi/4):
        r,i= filters.gabor(test, 0.2, theta=0)
        r=r.reshape(-1)
        i=i.reshape(-1)
        df_r[num]= r
        df_i[num]=i
        num+=1
    
        





