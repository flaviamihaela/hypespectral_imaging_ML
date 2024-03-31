
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

#define functions
def calculate_mean_masked_spectra(reflArray,osavi,osavi_threshold,ineq='>'):

    mean_masked_refl = np.zeros(reflArray.shape[2])

    for i in np.arange(reflArray.shape[2]):
        refl_band = reflArray[:,:,i]

        if ineq == '>':
            osavi_mask = ma.masked_where((osavi<=osavi_threshold) | (np.isnan(osavi)),osavi)
        elif ineq == '<':
            osavi_mask = ma.masked_where((osavi>=osavi_threshold) | (np.isnan(osavi)),osavi)   
        else:
            print('ERROR: Invalid inequality. Enter < or >')

        masked_refl = ma.MaskedArray(refl_band,mask=osavi_mask.mask)
        mean_masked_refl[i] = ma.mean(masked_refl)

    return mean_masked_refl

#START PROCESSING

#read .raw file
data_ref = envi.open(r"C:\Users\FlaviaMihaela\Desktop\PLNT_BRSNN_CHRGR_INOCL_06dai_06.hdr", r"C:\Users\FlaviaMihaela\Desktop\PLNT_BRSNN_CHRGR_INOCL_06dai_06.raw")
data= np.array(data_ref.load())

#resize image if needed
#data=data[1000:1800,500:1400,:]
#v1= imshow(data, (12,9,3), stretch=(0.03, 0.99), figsize= (5,5))

#calculate osavi
R800=data[:,:,28]
R670=data[:,:,17]
osavi=(R800-R670)/(R800+R670+0.16)
#osavi=np.divide((R800-R670),(R800+R670+0.16))
#v3= imshow(osavi, stretch=(0.72, 0.99), figsize= (10,10))
#plt.hist(osavi)
#plt.show()
m_otsu_t_osavi= filters.threshold_multiotsu(osavi)
print(m_otsu_t_osavi)
osavi_c=osavi


#set all pixels with OSAVI < 0.62 to nan, keeping only values > 0.71
osavi_c[osavi_c<0.71] = np.nan
mask_osavi=osavi_c<0.35
current_cmap = matplotlib.cm.get_cmap()
current_cmap.set_bad(color='0.8')
#plt.imshow(mask_osavi, cmap='gray', interpolation='nearest')
plt.imshow(osavi_c, cmap='gray', interpolation='nearest')

'''
#extract mean spectral signature of leaf after using OSAVI mask
w= [420, 440, 460, 480, 500, 520, 530, 540, 550, 560, 580, 590, 600, 610, 620, 630, 650, 670, 690, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 830, 860, 890, 900, 930, 960]
extr_osavi = calculate_mean_masked_spectra(data,osavi,0.34806257)
extr_osavi=scipy.signal.savgol_filter(extr_osavi, window_length=7, polyorder=5)
extr_osavi_df = pd.DataFrame()
extr_osavi_df['wavelength'] = w
extr_osavi_df['mean_refl_osavi'] = extr_osavi
#extr_osavi_df = extr_osavi_df.set_index('wavelength')
#extr_osavi_df.index.name=''
print(extr_osavi_df.head())
ax = plt.gca();
extr_osavi_df.plot(ax=ax,x='wavelength',y='mean_refl_osavi',color='green',kind='line',label='OSAVI',legend=True);
ax.set_title('Mean Spectra of Reflectance Masked by OSAVI')
ax.set_xlabel("Wavelength, nm"); ax.set_ylabel("Reflectance")
ax.grid('on'); 

'''

