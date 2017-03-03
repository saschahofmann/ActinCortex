import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ImageProcess as IM
import theoreticalModel as TM
import lmfit
import cv2
#Collect all filenames
Data = []
# Search in the current directory (./) and subdirectories for .tif files
for root, dirs, files in os.walk("."):        
    for file in files:
            if file.endswith(".tif"):
    #            try:
    #                int(file.split('_')[-1].split('.')[0])
    #            except:
    # Add the Path of .tif files to Data 
                    Data.append(os.path.join(root, file))

# Define Pixelsize, Cellmax and min sizen, bin size for averaging 
# and the length of the linescans
#TODO: include automated pixelsize recognition
pix_size = 101.61/512  #um/pix
cell_max_y = 25
cell_max_x = 25
cell_min = 10
linescan_length = 100
#bin_size =range(1, 251,1)
bin_size = 150
corr = 1.0015       # Magnification correction factor for chromatic shift
calc_dist = 1.     # Dist from peak for i_in and i_out calculation in um
sigma = 0.17
#Panda.DataFrame is tabular-like with column and row labelling possible ??
df_ALL = pd.DataFrame()    #????? 
"""
for filename in Data:
    #TODO: check whether original pic should be used for linescans  
    original = plt.imread(filename) # reads in .tif and .lsm as 
                                    # several greyscale images
       
        
    memb_original = original[:,:,1]          # membrane picture should be pic number 1
    actin_original = original[:,:, 0]        # actin cortex number 0
"""
for filename in Data:
    if filename.endswith("memb.tif"):
        memb_original = cv2.imread(filename,0)
    elif filename.endswith("actin.tif"):
        actin_original = cv2.imread(filename,0)

memb = IM.scale(memb_original)
actin = IM.scale(actin_original)
x_contour, y_contour = IM.smoothed_contour(memb, cell_max_x, cell_max_y, 
                                           cell_min, pix_size, 
                                           linescan_length, 1000)
plt.figure(2)
plt.imshow(memb, origin='lower',cmap = 'gray', interpolation = 'bilinear',vmin=0,vmax=255)
plt.plot(x_contour,y_contour)
plt.show()
plt.close()
memb_ls, actin_ls= IM.smooth_Linescan(memb_original, actin_original,
                                      x_contour, y_contour, linescan_length )

rad = np.arange(linescan_length)*pix_size

average_memb = IM.average_Linescan(memb_ls, 200)[2]
average_actin = IM.average_Linescan(actin_ls, 200)[2]

memb_amp, memb_mean, memb_sigma = IM.fitten(average_memb, rad)
actin_amp, actin_mean, actin_sigma = IM.fitten(average_actin, rad)
Intensity_in, Intensity_out = IM.get_Intensities(average_actin, 
                                                    rad, actin_mean,calc_dist)

pars = TM.get_parameters_default2(memb_mean[0], sigma, Intensity_in[0], Intensity_out[0])
fit = lmfit.minimize(TM.residual2, pars, kws = {'x_c': actin_mean, 'i_p': actin_amp})
i_c = fit.params["i_c"].value
h = fit.params["h"].value
y = TM.convolution(rad, h, i_c, memb_mean[0], sigma, Intensity_in[0], Intensity_out[0])
smalldelta =  sigma**2 / h * np.log((Intensity_out[0] -i_c)/(Intensity_in[0] - i_c))      
delta =  h/2 + smalldelta       
X_c = memb_mean[0] - delta + smalldelta
x = rad - X_c + h/2
from scipy.special import erf 
a = np.sqrt(1/(2*sigma**2))
integral = (i_c -Intensity_in[0])*0.5*(1- erf(-a*x))
#print integral
#print x , i_c -Intensity_in[0], X_c
#print (i_c -Intensity_in[0])*TM.cdf(np.float64(x), sigma)
cdf1 = (i_c -Intensity_in[0])*TM.cdf(rad -X_c + h/2 , sigma)
cdf2 = (Intensity_out[0] - i_c)*TM.cdf(rad- X_c -h/2, sigma)
plt.figure(1)
plt.plot(rad, average_memb,c  ="b")
plt.plot(rad, average_actin, c = "g")
plt.plot(rad, y, c = "r")
#plt.plot(rad, integral)
#plt.plot(rad, cdf2)
#plt.plot(rad, cdf1)
plt.show()

print h

"""
bin_size = range(1, len(memb_ls)+1, 20)
av_h = []
h_list =[]
plt.figure(1, figsize =(16,16))
for i in bin_size:
    average_memb = IM.average_Linescan(memb_ls, i, )
    average_actin = IM.average_Linescan(actin_ls, i, )
    #Get intra- and extracellular intensities
    memb_amp, memb_mean, memb_sigma = IM.fitten(average_memb, rad)
    actin_amp, actin_mean, actin_sigma = IM.fitten(average_actin, rad)
    Intensity_in, Intensity_out = IM.get_Intensities(average_actin, 
                                                    rad, actin_mean,calc_dist)
                          
    thick = []
    for j in range(len(average_memb)):
        x_m = memb_mean[j]
        i_in, i_out = Intensity_in[j], Intensity_out[j]
        pars2 = TM.get_parameters_default2(x_m, sigma, i_in, i_out)
        fit2 = lmfit.minimize(TM.residual2, pars2, kws={'x_c': actin_mean[j], 'i_p': actin_amp[j]})
        h = fit2.params["h"].value
        thick.append(h)   
    av_h.append(np.mean(thick))
    h_list.append(thick)
    plt.plot(i*np.ones(len(thick)), thick, c = "b", ls = "", marker = "+")
    
print h_list  
plt.title("Bin Size dependency of delta")
plt.plot(bin_size, av_h, 'o-', c = "r", label = "Bin Size dependency")
plt.xlabel('Bin Size')
plt.ylabel('h [um]')
plt.legend()
plt.savefig(filename.rstrip(".lsm")+"_"+str(np.max(bin_size)/len(bin_size))+'_binsteps_0_overlap.png')
plt.show()
"""

