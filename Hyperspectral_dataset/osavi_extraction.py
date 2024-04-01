# Import necessary libraries and modules
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

## Function to calculate mean spectral signatures for areas selected by an OSAVI threshold
def calculate_mean_masked_spectra(reflArray,osavi,osavi_threshold,ineq='>'):
    
    # Initialize an array to hold mean reflectance values for each spectral band
    mean_masked_refl = np.zeros(reflArray.shape[2])

    # Loop through each spectral band
    for i in np.arange(reflArray.shape[2]):
        refl_band = reflArray[:,:,i] # Extract the current band

        # Apply OSAVI mask based on the specified threshold and inequality
        if ineq == '>':
            osavi_mask = ma.masked_where((osavi<=osavi_threshold) | (np.isnan(osavi)),osavi)
        elif ineq == '<':
            osavi_mask = ma.masked_where((osavi>=osavi_threshold) | (np.isnan(osavi)),osavi)   
        else:
            print('ERROR: Invalid inequality. Enter < or >')

        # Apply the mask to the current band and calculate the mean of the unmasked values
        masked_refl = ma.MaskedArray(refl_band,mask=osavi_mask.mask)
        mean_masked_refl[i] = ma.mean(masked_refl)

    return mean_masked_refl

# START OF MAIN PROCESSING

# Load hyperspectral data from specified file paths
data_ref = envi.open(r"C:\Users\FlaviaMihaela\Desktop\PLNT_BRSNN_CHRGR_INOCL_06dai_06.hdr", r"C:\Users\FlaviaMihaela\Desktop\PLNT_BRSNN_CHRGR_INOCL_06dai_06.raw")
data= np.array(data_ref.load()) # Convert data to a NumPy array for processing

# Resize image if needed
#data=data[1000:1800,500:1400,:]
#v1= imshow(data, (12,9,3), stretch=(0.03, 0.99), figsize= (5,5))

# Calculate OSAVI using specific bands (R800 and R670)
R800=data[:,:,28]
R670=data[:,:,17]
osavi=(R800-R670)/(R800+R670+0.16)
#osavi=np.divide((R800-R670),(R800+R670+0.16))
#v3= imshow(osavi, stretch=(0.72, 0.99), figsize= (10,10))
#plt.hist(osavi)
#plt.show()

# Use Otsu's method to find an optimal threshold for OSAVI
m_otsu_t_osavi= filters.threshold_multiotsu(osavi)
print(m_otsu_t_osavi) # Print the thresholds found by Otsu's method

# Apply thresholding to OSAVI, setting values below 0.71 to NaN
osavi_c=osavi
osavi_c[osavi_c<0.35] = np.nan

# Create a mask for OSAVI values and set the color for NaN values in the colormap
mask_osavi=osavi_c>0.35
current_cmap = matplotlib.cm.get_cmap()
current_cmap.set_bad(color='0.8')
#plt.imshow(mask_osavi, cmap='gray', interpolation='nearest')

# Display the OSAVI image with applied thresholding
plt.imshow(osavi_c, cmap='gray', interpolation='nearest')


# Extract mean spectral signature of leaf after using OSAVI mask
w= [420, 440, 460, 480, 500, 520, 530, 540, 550, 560, 580, 590, 600, 610, 620, 630, 650, 670, 690, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 830, 860, 890, 900, 930, 960]
extr_osavi = calculate_mean_masked_spectra(data,osavi,0.35)
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

