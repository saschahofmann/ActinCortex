import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ImageProcess as IM
import theoreticalModel as TM
import lmfit
import cv2
from matplotlib_scalebar.scalebar import ScaleBar
#Collect all filenames
Data = []
# Search in the current directory (./) and subdirectories for .tif files
for root, dirs, files in os.walk("."):        
    for file in files:
            if file.endswith(".lsm"):
    #            try:
    #                int(file.split('_')[-1].split('.')[0])
    #            except:
    # Add the Path of .tif files to Data 
                    Data.append(os.path.join(root, file))

# Define Pixelsize, Cellmax and min sizen, bin size for averaging 
# and the length of the linescans
#TODO: include automated pixelsize recognition
pix_size = 40.65/512  # um/pix
cell_max_y = 25         # um
cell_max_x = 25         # um
cell_min = 10           # um
linescan_length = 100   # in pix
#bin_size =range(1, 251,1)
bin_size = 150
corr = 1.0015       # Magnification correction factor for chromatic shift
calc_dist = 1    # Dist from peak for i_in and i_out calculation in um
sigma = 0.250
#Panda.DataFrame is tabular-like with column and row labelling possible ??
df_ALL = pd.DataFrame()    #????? 

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
"""      
memb = IM.scale(memb_original)
actin = IM.scale(actin_original)
x_contour, y_contour = IM.smoothed_contour(memb, cell_max_x, cell_max_y, 
                                           cell_min, pix_size, 
                                           linescan_length, 1000)
# Linescans need to be done for the original, unscaled images
memb_ls, actin_ls, x_0, y_0= IM.smooth_Linescan(memb_original, actin_original,
                                      x_contour, y_contour, linescan_length )
r = np.sqrt((x_0 - (np.ones(x_0.shape).T*x_0[:,0]).T)**2 + (y_0 - (np.ones(y_0.shape).T*y_0[:,0]).T)**2)
rad = r * pix_size
plt.figure(2)
plt.imshow(memb, origin='lower',cmap = 'gray', interpolation = 'bilinear',vmin=0,vmax=255)
plt.plot(x_contour,y_contour, c = 'r', linewidth = '2')
#plt.xlim(180, 320)
#plt.ylim(200, 325)
#scalebar = ScaleBar(pix_size,'um', frameon = False, color = 'w', location= 4)
#plt.gca().add_artist(scalebar)
plt.show()
"""
plt.figure(3)
plt.imshow(actin_ls, origin='lower',cmap = 'gray', interpolation = 'bilinear',vmin=0,vmax=255)
plt.ylim(100, 400)
scalebar = ScaleBar(pix_size,'um', frameon = False, color = 'w', location= 4, height_fraction= 0.02, length_fraction = 0.4)
plt.gca().add_artist(scalebar)
plt.axvline(50, 0, 1, c = 'g', lw = 2 )
plt.axis('off')
plt.show()
"""



bin_size = 100

average_memb = IM.average_Linescan(memb_ls, bin_size)
average_actin = IM.average_Linescan(actin_ls, bin_size)
average_rad = IM.average_Linescan(rad,bin_size)

memb_amp, memb_mean, memb_sigma = IM.fitten(average_memb, average_rad)
actin_amp, actin_mean, actin_sigma = IM.fitten(average_actin, average_rad)
Intensity_in, Intensity_out = IM.get_Intensities(average_actin, 
                                                    average_rad, actin_mean,calc_dist)
h = []
for i in xrange(len(average_memb)):
    pars = TM.get_parameters_default(memb_mean[i], sigma, Intensity_in[i], Intensity_out[i])
    fit = lmfit.minimize(TM.residual, pars, kws = {'x_c': actin_mean[i], 'i_p': actin_amp[i]})
    i_c = fit.params["i_c"].value
    h.append(fit.params["h"].value)

print np.mean(h)
plt.figure(3)
for i in xrange(len(average_rad)):
    plt.plot(average_rad[i], average_actin[i])
plt.show()
"""
actin_max = np.argmax(average_actin)
memb_max = np.argmax(average_memb)
x_act2 = average_rad[actin_max-2: actin_max +3]
x_act = np.linspace(average_rad[actin_max-2], average_rad[actin_max+2], endpoint = True)
x_memb = np.linspace(average_rad[memb_max -2],average_rad[memb_max +2], endpoint = True)
x_memb2 = average_rad[memb_max -2: memb_max + 3]
actin_gauss = IM.gauss(x_act, actin_amp[0], actin_mean[0], actin_sigma[0])
memb_gauss = IM.gauss(x_memb, memb_amp[0], memb_mean[0], memb_sigma[0])
actin_int_in = np.argmin(abs(average_rad - (actin_mean[0] -calc_dist)))
int_in =  average_actin[actin_int_in -10: actin_int_in]
actin_int_out = np.argmin(abs(average_rad - actin_mean[0] - calc_dist))
int_out = average_actin[actin_int_out: actin_int_out + 10]

#gauss_memb = IM.gauss()
plt.figure(1)
plt.ylabel('Intensity', fontsize = 18)
plt.xlabel(r'Position [$\mu m$]', fontsize = 18)
#plt.plot(average_rad, average_memb,c  ="b", label = 'Plasmamembrane')
plt.plot(average_rad, average_actin, c = "g", label = 'Actin Cortex')

x = np.linspace(average_rad[0], average_rad[-1], 1000)
y = TM.convolution(x, h, i_c, memb_mean[0], sigma, Intensity_in[0], Intensity_out[0])
plt.plot(x, y, c = "r", label = 'Convolution-Model')
#plt.plot(x_act, actin_gauss,  c = "r", lw = 2 )
#plt.plot(x_memb, memb_gauss, c = 'r', lw = 2)
#plt.plot(x_memb2, average_memb[memb_max -2: memb_max + 3], marker = 'x', c = 'black', ls = '', ms = 8)
#plt.plot(x_act2, average_actin[actin_max-2: actin_max +3], marker = 'x', c = 'black', ls = '',  ms = 8)
#plt.plot(average_rad[actin_int_in -10: actin_int_in], int_in, markersize= 10, c = '#20B2AA', ls = '', marker = 'o')
#plt.plot(average_rad[actin_int_out: actin_int_out +10], int_out, markersize= 10, c = '#20B2AA', ls = '', marker = 'o')
#plt.axhline(Intensity_in, 0, average_rad[actin_int_in]/ average_rad[-1], ls = '--', c = 'gray')
#plt.axhline(Intensity_out, 0, average_rad[actin_int_out]/ average_rad[-1], ls = '--', c ='gray')
#plt.axvline(memb_mean, -5, memb_amp[0]/200, ls = '--', c= 'gray')                                     

#plt.yticks(list(plt.yticks()[0]) +Intensity_in, list(str(plt.yticks()[1]))+  ['i_in'])

plt.ylim(-5,200)
plt.xlim(0,15)
plt.legend(loc = 2)

plt.show()

print h
print i_c
"""

bin_size = range(10, 300, 10)
av_h = []
h_list =[]
plt.figure(1, figsize =(16,16))
i_c = []
for i in bin_size:
    
    average_memb = IM.average_Linescan(memb_ls, i, )
    average_actin = IM.average_Linescan(actin_ls, i, )
    average_rad = IM.average_Linescan(rad, i,)
    #plt.plot(average_rad[499], average_memb[499])
    #plt.show()
    #Get intra- and extracellular intensities
    memb_amp, memb_mean, memb_sigma = IM.fitten(average_memb, average_rad)
    actin_amp, actin_mean, actin_sigma = IM.fitten(average_actin,average_rad)
    Intensity_in, Intensity_out = IM.get_Intensities(average_actin, 
                                                    average_rad, actin_mean,calc_dist)  
    thick = []
    for j in range(len(average_memb)):
        x_m = memb_mean[j]
        i_in, i_out = Intensity_in[j], Intensity_out[j]
        pars2 = TM.get_parameters_default(x_m, sigma, i_in, i_out)
        fit2 = lmfit.minimize(TM.residual, pars2, kws={'x_c': actin_mean[j], 'i_p': actin_amp[j]})
        h = fit2.params["h"].value
        thick.append(h)   
        i_c.append(fit2.params["i_c"].value)
    av_h.append(np.mean(thick))
    h_list.append(thick)
    plt.plot(i*np.ones(len(thick)), thick, c = "b", ls = "", marker = "+", ms = 10)
    
plt.title("Bin Size dependency of delta")
plt.plot(bin_size, av_h, 'o-', c = "r", label = "Mean Average Thickness")
plt.xlabel('Bin Size', fontsize= 22)
plt.ylabel('h [um]', fontsize= 22)
plt.legend()
plt.savefig(filename.rstrip(".tif")+"_"+str(np.max(bin_size)/len(bin_size))+'_binsteps_0_overlap.png')
plt.show()

