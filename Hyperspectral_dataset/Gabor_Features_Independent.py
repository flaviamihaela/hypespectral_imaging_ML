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
from cv2 import filter2D as _do_convolve

# Function to calculate the mean spectral signature for masked regions based on a threshold
def calculate_mean_masked_spectra(reflArray,cop,cop_threshold,ineq='>'):

    mean_masked_refl = np.zeros(reflArray.shape[2]) # Initialize array for mean spectral values

    # Iterate through each spectral band
    for i in np.arange(reflArray.shape[2]):
        refl_band = reflArray[:,:,i] # Extract the current band

         # Apply mask based on the threshold and condition (greater or less than)
        if ineq == '>':
            cop_mask = ma.masked_where((cop<=cop_threshold) | (np.isnan(cop)),cop)
        elif ineq == '<':
            cop_mask = ma.masked_where((cop>=cop_threshold) | (np.isnan(cop)),cop)   
        else:
            print('ERROR: Invalid inequality. Enter < or >')

        # Calculate mean reflectance of the masked band
        masked_refl = ma.MaskedArray(refl_band,mask=cop_mask.mask)
        mean_masked_refl[i] = ma.mean(masked_refl)

    return mean_masked_refl

# Function for Standard Normal Variate (SNV) transformation
def snv(input_data):
    data_snv = np.zeros_like(input_data) # Initialize array for SNV data
    # Apply SNV transformation to each sample
    for i in range(data_snv.shape[0]):
        data_snv[i] = (input_data[i] - np.mean(input_data)) / np.std(input_data)
    return (data_snv)

# Function to apply a mask across all spectral bands of a data array
def masked_spectra(reflArray, mask):
    
    M = np.zeros(reflArray.shape) # Initialize array for masked data

    # Apply the mask to each spectral band
    for i in np.arange(reflArray.shape[2]):
        refl_band = reflArray[:,:,i]
        M_band =M[:,:,i]
        M_band[mask] = refl_band[mask] # Apply mask to the current band
    
    return M


# START OF MAIN PROCESSING

# Load hyperspectral data from specified file paths
data_ref = envi.open(r"C:\Users\FlaviaMihaela\Desktop\PLNT_BRSNN_CHRGR_INOCL_21dai_06.hdr", r"C:\Users\FlaviaMihaela\Desktop\PLNT_BRSNN_CHRGR_INOCL_21dai_06.raw")
data= np.array(data_ref.load()) # Convert data to a NumPy array

# Apply Laplacian-Gaussian filter to enhance features and apply thresholding
copy_data=data[:,:,25] # Select a specific band for processing
#plt.hist(copy_data)
#plt.show()
copy_data_log=ndimage.gaussian_laplace(copy_data, sigma=11) # Apply LoG filter
#plt.imshow(copy_data,cmap='gray', interpolation='nearest')
thres = np.absolute(copy_data_log).mean() * 0.75 # Determine threshold for feature detection'

# Detect features using zero-crossing method
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
output=output.astype('bool') # Convert output to a boolean array for masking
sel = np.zeros_like(copy_data)
sel[output] = copy_data[output] # Apply the detected features as a mask
#plt.hist(sel)
#plt.show()


#Multiple thresholding using otsu's method
#m_otsu_t= filters.threshold_multiotsu(sel)
#regions = np.digitize(copy_data, bins=m_otsu_t)
#plt.imshow(regions, cmap='Accent')
#plt.show()
#print(m_otsu_t)


# Obtain a mask based on a specific threshold and apply it to the data
spare_data=copy_data
spare_data[spare_data<52.29492] = np.nan # Set values below threshold to NaN
mask_log=(copy_data>52.29492) # Create a binary mask for values above the threshold
#viewer = skimage.viewer.ImageViewer(mask_log)
#viewer.show()
sel2 = np.zeros_like(copy_data)
sel2[mask_log] = copy_data[mask_log] # Apply the mask
#plt.imshow(sel2, cmap='gray', interpolation='nearest')


# Obtain the segmented 3D image by applying the mask on all bands
m_data=masked_spectra(data,mask_log)
#v1= imshow(m_data, (12,9,3), stretch=(0.03, 0.99), figsize= (5,5))
#plt.show()


# Generate a Gabor filter bank and apply it to each spectral band
num=1
df_r=pd.DataFrame() # DataFrame to store real parts of Gabor-filtered images
df_i=pd.DataFrame() # DataFrame to store imaginary parts of Gabor-filtered images
frequency=0.2 # Frequency for Gabor filter
for i in np.arange(m_data.shape[2]):
    test = m_data[:,:,i] # Extract a spectral band
    # Iterate over angles for Gabor filter
    for theta in np.arange(0, np.pi, np.pi/4):
        # Apply Gabor filter
        r,i= filters.gabor(test, 0.2, theta=0)
        r=r.reshape(-1) # Flatten the real part
        i=i.reshape(-1) # Flatten the imaginary part
        df_r[num]= r # Store the real part in DataFrame
        df_i[num]=i # Store the imaginary part in DataFrame
        num+=1
    
        





