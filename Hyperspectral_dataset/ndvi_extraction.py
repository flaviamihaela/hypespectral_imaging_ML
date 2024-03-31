#Extract spectral signature with NDVI mask

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

#define functions
def calculate_mean_masked_spectra(reflArray,ndvi,ndvi_threshold,ineq='>'):

    mean_masked_refl = np.zeros(reflArray.shape[2])

    for i in np.arange(reflArray.shape[2]):
        refl_band = reflArray[:,:,i]

        if ineq == '>':
            ndvi_mask = ma.masked_where((ndvi<=ndvi_threshold) | (np.isnan(ndvi)),ndvi)
        elif ineq == '<':
            ndvi_mask = ma.masked_where((ndvi>=ndvi_threshold) | (np.isnan(ndvi)),ndvi)   
        else:
            print('ERROR: Invalid inequality. Enter < or >')

        masked_refl = ma.MaskedArray(refl_band,mask=ndvi_mask.mask)
        mean_masked_refl[i] = ma.mean(masked_refl)

    return mean_masked_refl


#START PROCESSING

#read .raw file
data_ref = envi.open(r"PLNT_BRSNN_CHRGR_INOCL_24dai_06.hdr", r"PLNT_BRSNN_CHRGR_INOCL_24dai_06.raw")
data= np.array(data_ref.load())


#calculate ndvi
red = data[:,:,17]
nir= data[:,:,28]
ndvi=(nir-red)/(nir+red+0.0001)
v3= imshow(ndvi, cmap='gray', interpolation='nearest')
print(np.amax(ndvi))
print(np.amin(ndvi))
#plt.hist(ndvi)
plt.show()
#make copy of ndvi for thresholding
ndvi_c = ((nir-red)/(nir+red+0.00001))
#apply otsu multithresholding to identify threshold value for NDVI
m_otsu_t= filters.threshold_multiotsu(ndvi_c)
print(m_otsu_t)
#values:0.84657586

#set all pixels with NDVI < 0.84657586 to nan, keeping only values > 0.818
ndvi_c[ndvi_c<0.15] = np.nan
mask_ndvi=ndvi_c>0.15
#print(ndvi_c.dtype)
current_cmap = matplotlib.cm.get_cmap()
current_cmap.set_bad(color='0.8')
plt.imshow(ndvi_c)
#print('Mean NDVI < 0.84657586:',round(np.nanmean(ndvi_c),2))

'''
#extract mean spectral signature of leaf after using NDVI mask
w= [420, 440, 460, 480, 500, 520, 530, 540, 550, 560, 580, 590, 600, 610, 620, 630, 650, 670, 690, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 830, 860, 890, 900, 930, 960]
extr_ndvi = calculate_mean_masked_spectra(data,ndvi,0.84657586)
extr_ndvi_df = pd.DataFrame()
extr_ndvi_df['wavelength'] = w
extr_ndvi_df['mean_refl_ndvi'] = (extr_ndvi/255)*100
#extr_ndvi_df = extr_ndvi_df.set_index('wavelength')
#extr_ndvi_df.index.name=''
print(extr_ndvi_df.head())
ax = plt.gca();
extr_ndvi_df.plot(ax=ax,x='wavelength',y='mean_refl_ndvi',color='green',kind='line',label='NDVI > 0.818',legend=True);
ax.set_title('Mean Spectra of Reflectance Masked by NDVI')
ax.set_xlabel("Wavelength, nm"); ax.set_ylabel("Reflectance (%)")
ax.grid('on'); 
'''