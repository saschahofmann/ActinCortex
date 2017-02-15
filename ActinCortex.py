import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ImageProcess as IM
import theoreticalModel as TM
import lmfit


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
pix_size = 64.02/512  #um/pix
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

for filename in Data:
    #TODO: check whether original pic should be used for linescans  
    original = plt.imread(filename) # reads in .tif and .lsm as 
                                    # several greyscale images
    memb = original[:,:,1]          # membrane picture should be pic number 1
    actin = original[:,:, 0]        # actin cortex number 0
    memb = IM.scale(memb)
    actin = IM.scale(actin)
    memb_ls, actin_ls= IM.smooth_Linescan(memb, actin, cell_max_x, cell_max_y, 
                                          cell_min, linescan_length, pix_size, 600 )
    rad = np.arange(linescan_length)*pix_size
    """
    average_memb = IM.average_Linescan(memb_ls, bin_size, 100)[3]
    average_actin = IM.average_Linescan(actin_ls, bin_size, 100)[3]
    memb_amp, memb_mean, memb_sigma = IM.fitten(average_memb, rad)
    actin_amp, actin_mean, actin_sigma = IM.fitten(average_actin, rad)
    Intensity_in, Intensity_out = IM.get_Intensities(average_actin, 
                                                        rad, actin_mean,calc_dist)
    pars = TM.get_parameters_default2(memb_mean[0], sigma, Intensity_in[0], Intensity_out[0])
    fit = lmfit.minimize(TM.residual2, pars, kws = {'x_c': actin_mean, 'i_p': actin_amp})
    y = TM.convolution(rad, fit.params["h"].value, fit.params["i_c"].value, memb_mean[0], sigma, Intensity_in[0], Intensity_out[0])
    plt.figure(1)
    plt.plot(rad, average_memb)
    plt.plot(rad, average_actin)
    plt.plot(rad, y)
    plt.show()
"""
    bin_size = range(1, len(memb_ls)+1, 20)
    av_h = []
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
        plt.plot(i*np.ones(len(thick)), thick, c = "b", ls = "", marker = "+")
        
        
    plt.title("Bin Size dependency of delta")
    plt.plot(bin_size, av_h, 'o-', c = "r", label = "Bin Size dependency")
    plt.xlabel('Bin Size')
    plt.ylabel('h [um]')
    plt.legend()
    plt.savefig(filename.rstrip(".lsm")+"_"+str(np.max(bin_size)/len(bin_size))+'_binsteps_0_overlap.png')
    plt.show()
      