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

# Define functions for processing

# Function to calculate mean spectral signature for masked regions
def calculate_mean_masked_spectra(reflArray,cop,cop_threshold,ineq='>'):
    
    # Initialize array to hold mean spectral values
    mean_masked_refl = np.zeros(reflArray.shape[2]) 

    # Iterate through each spectral band
    for i in np.arange(reflArray.shape[2]):
        # Get the current band
        refl_band = reflArray[:,:,i]

        # Apply mask based on condition and threshold
        if ineq == '>':
            cop_mask = ma.masked_where((cop<=cop_threshold) | (np.isnan(cop)),cop)
        elif ineq == '<':
            cop_mask = ma.masked_where((cop>=cop_threshold) | (np.isnan(cop)),cop)   
        else:
            print('ERROR: Invalid inequality. Enter < or >')

        # Apply the mask to the current band and calculate the mean
        masked_refl = ma.MaskedArray(refl_band,mask=cop_mask.mask)
        mean_masked_refl[i] = ma.mean(masked_refl)

    return mean_masked_refl

# Function for Standard Normal Variate (SNV) transformation
def snv(input_data):

    # Initialize array for SNV data
    data_snv = np.zeros_like(input_data)
    # Apply SNV to each sample
    for i in range(data_snv.shape[0]):
        data_snv[i] = (input_data[i] - np.mean(input_data)) / np.std(input_data)
    return (data_snv)

# START OF MAIN PROCESSING

# Load the hyperspectral data from specified file paths
data_ref = envi.open(r"C:\Users\FlaviaMihaela\Desktop\PLNT_BRSNN_CHRGR_INOCL_06dai_06.hdr", r"C:\Users\FlaviaMihaela\Desktop\PLNT_BRSNN_CHRGR_INOCL_06dai_06.raw")

# Convert data to NumPy array for processing
data= np.array(data_ref.load())

# Resize image if needed
#data=data[1000:1800,500:1400,:]

# Apply Laplacian of Gaussian (LoG) filter to enhance edges
copy_data=data[:,:,25]
#plt.hist(copy_data)
#plt.show()
copy_data_log=ndimage.gaussian_laplace(copy_data, sigma=11)
#plt.imshow(copy_data,cmap='gray', interpolation='nearest')

# Thresholding to create binary mask from one of the bands
thres = np.absolute(copy_data_log).mean() * 0.75
output = np.zeros(copy_data_log.shape)
w = output.shape[1]
h = output.shape[0]

# Apply thresholding to identify edges and obtain an edge image
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


# Multiple thresholding using otsu's method
m_otsu_t= filters.threshold_multiotsu(sel)
#regions = np.digitize(copy_data, bins=m_otsu_t)
#plt.imshow(regions, cmap='Accent')
#plt.show()
#print(m_otsu_t)
#values:52.29492, 50.302734


# Obtaining the mask
#spare_data=copy_data
#spare_data[spare_data<50.302734] = np.nan

# Mask application and spectral signature extraction
mask_log=(copy_data<50.302734) # Create a mask based on found threshold from edge image
viewer = skimage.viewer.ImageViewer(mask_log)
viewer.show()
sel2 = np.zeros_like(copy_data)
sel2[mask_log] = copy_data[mask_log]
#plt.imshow(sel2, cmap='gray', interpolation='nearest')

# Extract mean spectral signature of hyperspectral image of leaf 
w= [420, 440, 460, 480, 500, 520, 530, 540, 550, 560, 580, 590, 600, 610, 620, 630, 650, 670, 690, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 830, 860, 890, 900, 930, 960]

extr_log = calculate_mean_masked_spectra(data,copy_data,50.302734)
extr_log=scipy.signal.savgol_filter(extr_log, window_length=7, polyorder=5)
#extr_log=snv(extr_log)
extr_log_df = pd.DataFrame()
extr_log_df['wavelength'] = w
extr_log_df['mean_refl_log'] = extr_log
#extr_ndvi_df = extr_ndvi_df.set_index('wavelength')
#extr_ndvi_df.index.name=''
print(extr_log_df.head())
ax = plt.gca();
extr_log_df.plot(ax=ax,x='wavelength',y='mean_refl_log',color='green',kind='line',label='LoG',legend=True);
ax.set_title('Mean Spectra of Reflectance Masked by LoG')
ax.set_xlabel("Wavelength, nm"); ax.set_ylabel("Reflectance")
ax.grid('on'); 