import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ImageProcess as IM
import theoreticalModel as TM
import lmfit
from scipy.interpolate import RectBivariateSpline

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
bin_size = 200
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
    contour, locus = IM.contour_1(memb, cell_max_x, cell_max_y, cell_min, pix_size, linescan_length)
    contour2 = IM.contour_2(contour, locus)
    memb_ls, radius = IM.smooth_Linescan(memb, contour2, linescan_length, pix_size)
    

    dx = np.diff(contour2[0])
    dy = np.diff(contour2[1])
    dx = np.append(dx, contour2[0][0]-contour2[0][-1])
    dy = np.append(dy,contour2[1][0]-contour2[1][-1])
    derivative = dy/dx
    normal = -1.0/derivative
    n = contour2[1] - normal * contour2[0]
    x = np.linspace(230, 240, 500)
    line = normal[0]* x + n[0]
    
    memb_interpol = RectBivariateSpline(np.arange(memb.shape[0]), np.arange(memb.shape[1]), memb)
    linescan = memb_interpol.ev(x, line)
    plt.figure(1)
    plt.imshow(memb, origin='lower',cmap = 'gray', interpolation = 'bilinear',vmin=0,vmax=255)
    plt.plot(contour2[0], contour2[1],c = "g")
    plt.plot(contour[:,0], contour[:,1], c = "violet")
    plt.plot(x, line)
    plt.plot(contour2[0][0], contour2[1][0], c = "r", marker = "+")
    #plt.plot(contour[:, 0], contour[:,1])
    #plt.plot(contour2[0],contour2[1])
    plt.figure(2)
    plt.plot(np.arange(len(x)), linescan)
    plt.show()
    
    
    """
    memb_ls, actin_ls, radius = IM.linescan(memb, actin, cell_max_x,cell_max_y, 
                                            cell_min, pix_size,linescan_length)
    plt.imshow(memb_ls,origin='lower',cmap = 'gray', interpolation = 'bilinear',vmin=0,vmax=255)
    plt.show()

    bin_size = range(1, len(memb_ls)+1, 40)
    av_h = []
    for i in bin_size:
        average_memb = IM.average_Linescan(memb_ls, i, )
        average_actin = IM.average_Linescan(actin_ls, i, )
        rad = IM.average_Linescan(radius, i, )
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
    
      
    plt.figure(1, figsize =(16,16))
    plt.title("Bin Size dependency of delta")
    plt.plot(bin_size, av_h, 'o-', label = "Bin Size dependency")
    plt.xlabel('Bin Size')
    plt.ylabel('h [um]')
    plt.legend()
    plt.savefig(filename.rstrip(".lsm")+"_"+str(np.max(bin_size)/len(bin_size))+'_binsteps_0_overlap.png')
    plt.show()

    # Average bin_size linescans
    average_memb = IM.average_Linescan(memb_ls, bin_size)
    average_actin = IM.average_Linescan(actin_ls, bin_size)
    rad = IM.average_Linescan(radius, bin_size)
    # Get the gauss parameters of the peak
    memb_amp, memb_mean, memb_sigma = IM.fitten(average_memb, rad)
    actin_amp, actin_mean, actin_sigma = IM.fitten(average_actin, rad)
    #Calculate delta
    average_delta = []
    for i in range(len(memb_mean)):
        average_delta.append(memb_mean[i]- actin_mean[i])
    #Get intra- and extracellular intensities
    Intensity_in, Intensity_out = IM.get_Intensities(average_actin, 
                                                     rad, actin_mean,calc_dist)
    num = 1
    x_m = memb_mean[num]
    i_in, i_out = Intensity_in[num], Intensity_out[num]
    
    pars2 = TM.get_parameters_default2(x_m, sigma, i_in, i_out)
    fit2 = lmfit.minimize(TM.residual2, pars2, kws={'x_c': actin_mean[num], 'i_p': actin_amp[num]})#, method = "powell")
    
    print fit2.params["h"]
    print fit2.params["i_c"]
    
    h = fit2.params["h"].value
    i_c = fit2.params["i_c"].value
    x = rad[num] # np.linspace(9,9.5,300)      
    x_c = TM.checker(h, i_c, x_m, sigma, i_in, i_out)
    i_p = TM.convolution(x_c, h, i_c, x_m, sigma, i_in, i_out)
    y_model = TM.convolution(x, h, i_c, x_m, sigma, i_in, i_out)
    plt.plot(x, average_memb[num])
    plt.plot(rad[num], average_actin[num])   
    plt.plot(x, y_model )
    plt.axvline(x_c)
    plt.axhline(i_p)
    plt.show()
    
   
    
    #Get fit parameters for theoretical model
    pars = TM.get_parameters_default(average_delta[0], memb_mean[0], 1.359,
                                      Intensity_in[0], Intensity_out[0])  
    fit = lmfit.minimize(TM.residual, pars, args=(rad[0],), 
                         kws={'linescan':average_actin[0]} )
    print Intensity_in
    print fit.params["h"]
    print fit.params["i_c"]
    plt.plot(rad[0], average_actin[0])
    plt.plot(rad[0], TM.model(fit.params, rad[0]))
    plt.show()
    
    
    Cells = IM.findContour(memb, cell_max_x, cell_max_y, cell_min, pix_size)
    #Plotting of the contour
    nullfmt   = NullFormatter() 
    plt.figure(3,figsize=(8,8))
    plt.imshow(memb,cmap = 'gray', interpolation = 'bilinear',vmin=0,vmax=255)
    for i in range(len(Cells)):
        contour = Cells[i]['contour']
        plt.plot(contour[:,0],contour[:,1],label='Size='+str(Cells[i]['area_um']))
        plt.legend(loc='upper right', fancybox=True, framealpha=0.5)
    plt.axis('off')
    plt.savefig(filename.rstrip(".lsm")+'_Contours.png')
    plt.show()
    plt.close()  
    

    
    #Plot
    plt.figure(1, figsize= (16,16))
    for i in range(len(average_memb)):
        plt.subplot(3,2,i+1)
        plt.title(i)
        #Membranlinescanplot
        plt.plot(rad[i],average_memb[i],ls = '', marker = '.')
        gauss = IM.gauss(rad[i],memb_amp[i], memb_mean[i], memb_sigma[i])
        plt.plot(rad[i],gauss)
        # Actinlinescanplot
        plt.plot(rad[i],average_actin[i],c = "r", ls = '', marker = '.')
        gauss2 = IM.gauss(rad[i],actin_amp[i], actin_mean[i],actin_sigma[i])
        plt.plot(rad[i],gauss2)
        plt.xlabel('Rho [um]')
        plt.ylabel('Linescan (Intensity)')
    plt.savefig(filename.rstrip(".lsm")+'_average_Linescan.png')
    plt.show()   
    plt.close() 
    """ 
    