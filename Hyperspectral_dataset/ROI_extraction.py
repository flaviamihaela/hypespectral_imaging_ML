#Bounding Box- necrotic pixels extraction

#import libraries and modules
import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import skimage
from skimage import filters
import numpy.ma as ma
import pandas as pd
from skimage import filters
import scipy
from skimage import viewer

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


def zeros(a):
        return (np.all(a == 0))
    
def sg(a):
        return scipy.signal.savgol_filter(a, window_length=7, polyorder=5)

#START PROCESSING

#read .raw file
data_ref = envi.open(r"C:\Users\FlaviaMihaela\Desktop\PLNT_BRSNN_CHRGR_INOCL_24dai_06.hdr", r"C:\Users\FlaviaMihaela\Desktop\PLNT_BRSNN_CHRGR_INOCL_24dai_06.raw")
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


#Obtaining the mask and masking the data
#spare_data=copy_data
#spare_data[spare_data<52.29492] = np.nan
mask_log=(copy_data>52.29492)
viewer = skimage.viewer.ImageViewer(mask_log)
viewer.show()

#Obtain the segmented 3D datacube by applying the mask on all bands
m_data=masked_spectra(data,mask_log)
#Apply gaussian filter for noise to prepare it for SVI
m_data=ndimage.gaussian_filter(m_data, sigma=2)
#plt.imshow(m_data[:,:,0], cmap='gray', interpolation='nearest')

#Apply LLSI on the segmented datacube to obtain the LLSI matrix
#R720 = m_data[:,:,20]
#R530= m_data[:,:,6]
#R830= m_data[:,:,29]
#llsi=((R720-R530)/(R720+R530+0.00001))-R830
#otsu_t_llsi= filters.threshold_otsu(llsi)
#print(otsu_t_llsi)
#llsi[llsi>otsu_t_llsi] = np.nan
#mask_llsi=llsi<otsu_t_llsi
#Obtain ROI cube
#m_data_llsi=masked_spectra(m_data,mask_llsi)


#Apply CTR1 on the segmented datacube to obtain the CTR1 matrix
R700=m_data[:,:,19]
R420=m_data[:,:,0]
ctr1=R700/(R420+0.00001)
otsu_t_ctr1= filters.threshold_otsu(ctr1)
print(otsu_t_ctr1)
ctr1[ctr1>otsu_t_ctr1] = np.nan
mask_ctr1=ctr1<otsu_t_ctr1
#Obtain ROI cube
m_data_ctr1=masked_spectra(m_data,mask_ctr1)
#
m_data_ctr1=m_data_ctr1[1500:1650,1000:1150,:]
k=m_data_ctr1[:,:,24]
#plt.imshow(k, cmap='gray', interpolation='nearest')
#np.save('DAI03.npy', m_data_ctr1)


