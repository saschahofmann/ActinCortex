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
            if file.endswith(".tif"):
    #            try:
    #                int(file.split('_')[-1].split('.')[0])
    #            except:
    # Add the Path of .tif files to Data 
                    Data.append(os.path.join(root, file))

# Define Pixelsize, Cellmax and min sizen, bin size for averaging 
# and the length of the linescans
#TODO: include automated pixelsize recognition
pix_size = 101.61/512  # um/pix
cell_max_y = 25         # um
cell_max_x = 25         # um
cell_min = 10           # um
linescan_length = 100   # in pix
#bin_size =range(1, 251,1)
bin_size = 150
corr = 1.0015       # Magnification correction factor for chromatic shift
calc_dist = 2    # Dist from peak for i_in and i_out calculation in um
sigma = 0.250
#Panda.DataFrame is tabular-like with column and row labelling possible ??
df_ALL = pd.DataFrame()    #????? 
"""
for filename in Data:
    #TODO: check whether original pic should be used for linescans  
    original = plt.imread(filename) # reads in .tif and .lsm as 
                                    # several greyscale images
       
        
    memb_original = original[:,:,0]          # membrane picture should be pic number 1
    actin_original = original[:,:, 1]        # actin cortex number 0
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

memb_ls, actin_ls, x_0, y_0= IM.smooth_Linescan(memb_original, actin_original,
                                      x_contour, y_contour, linescan_length )
r = np.sqrt((x_0 - (np.ones(x_0.shape).T*x_0[:,0]).T)**2 + (y_0 - (np.ones(y_0.shape).T*y_0[:,0]).T)**2)
rad = r * pix_size
#plt.figure(2)
#plt.imshow(memb, origin='lower',cmap = 'gray', interpolation = 'bilinear',vmin=0,vmax=255)
#plt.plot(x_contour,y_contour, c = 'r', linewidth = '2')
#plt.xlim(180, 320)
#plt.ylim(200, 325)
#scalebar = ScaleBar(pix_size,'um', frameon = False, color = 'w', location= 4)
#plt.gca().add_artist(scalebar)
#plt.plot()

plt.figure(3)
plt.imshow(actin_ls, origin='lower',cmap = 'gray', interpolation = 'bilinear',vmin=0,vmax=255)
#plt.ylim(100, 400)
scalebar = ScaleBar(pix_size,'um', frameon = False, color = 'w', location= 4)
plt.axis('off')
plt.show()



"""
bin_size = 100

average_memb = IM.average_Linescan(memb_ls, bin_size)[8]
average_actin = IM.average_Linescan(actin_ls, bin_size)[8]
average_rad = IM.average_Linescan(rad,bin_size)[8]

memb_amp, memb_mean, memb_sigma = IM.fitten(average_memb, average_rad)
actin_amp, actin_mean, actin_sigma = IM.fitten(average_actin, average_rad)
Intensity_in, Intensity_out = IM.get_Intensities(average_actin, 
                                                    average_rad, actin_mean,calc_dist)

pars = TM.get_parameters_default(memb_mean[0], sigma, Intensity_in[0], Intensity_out[0])
fit = lmfit.minimize(TM.residual, pars, kws = {'x_c': actin_mean[0], 'i_p': actin_amp[0]})
i_c = fit.params["i_c"].value
h = fit.params["h"].value
y = TM.convolution(average_rad, h, i_c, memb_mean[0], sigma, Intensity_in[0], Intensity_out[0])

plt.figure(1)
plt.plot(average_rad, average_memb,c  ="b")
plt.plot(average_rad, average_actin, c = "g")
#plt.plot(average_rad, IM.gauss(average_rad, actin_amp[0], actin_mean[0], actin_sigma[0]))
plt.plot(average_rad, y, c = "r")
plt.show()

print h
print i_c
"""
bin_size = range(2, 300, 10)
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
    plt.plot(i*np.ones(len(thick)), thick, c = "b", ls = "", marker = "+")
    
print h_list  
print i_c
plt.title("Bin Size dependency of delta")
plt.plot(bin_size, av_h, 'o-', c = "r", label = "Bin Size dependency")
plt.xlabel('Bin Size')
plt.ylabel('h [um]')
plt.legend()
plt.savefig(filename.rstrip(".tif")+"_"+str(np.max(bin_size)/len(bin_size))+'_binsteps_0_overlap.png')
plt.show()


