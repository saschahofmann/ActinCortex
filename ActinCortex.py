""" 
Actin Cortex Thickness
https://github.com/saschahofmann/ActinCortex
@author: Sascha Hofmann, Biotec TU Dresden, Guck Lab
Following the paper 'Monitoring Actin Cortex Thickness in Live Cells' by 
Andrew G. Clark, Kai Dierkes and Ewa K. Paluch
Created: 20 April 2017
ActinCortex_v1
This program opens multi-color cell images in either tif or lsm format located 
in the directory. From these images it obtains an average Actin Cortex 
thickness.

REQUIREMENTS:
Optimally the images are corrected for chromatic aberration e.g. with the matlab
function 'chromatic_shift_corrector' by Andrew G. Clark. While writting images 
with a 512x512 size and a pixel size of ~70nm were used, for different 
properties some more parameters may needed to be changed.

INPUTS:
You need to put in the pixel size which can be found by opening the image with 
e.g. ImageJ, an automatic in-read of the Exif data will maybe added in a later 
version. Choose the data format and check the standard deviation of the 
microscope, it can be obtained from the point spread function. You can choose 
which plots you would like to see, this is mainly for debugging purposes. 
LSM images should consist of two channel images with the membrane image in 
the second and the cortex in the second channel (can be changed shortly after 
the imread command). At the moment only one single cell should be observed.


"""

# Additional Packages
import os
import numpy as np
import matplotlib.pyplot as plt
import lmfit
import cv2

#Functions for Image Analysis and Fitting
import ImageProcess as IM
import theoreticalModel as TM

# INPUT-PARAMETERS:
# Define Pixelsize, bin size for averaging, threshold and the length of the 
# linescans. The default is chosen for LSM700 images taken with the 63x oil 
# immersion objective. Sigma needs to be chose according to the PSF of the 
# Microscope depending on the used Laser
# TODO: include automated pixelsize recognition
pix_size = 101.61/512    # um/pix
linescan_length = 100   # pix 
threshold = 60          # Threshold for finding the contour/cell
sigma = 0.250           # um
bin_size = 150
calc_dist = 1           # Distance from peak to calculate intra- and 
                        # extracellular Intensity in um
n = 1000                # Points in the contour = Number of Linescans
#Choose file format (for tif the channels have to be split and named 'actin' 
#  and 'memb':
file_format = ".tif"    # ".lsm" or ".tif"
# Choose maximal and minimal Cellsize to filter the contours
cell_max_y = 25         # um
cell_max_x = 25         # um
cell_min = 10           # um
# Choose which image plots should be shown:
show_thresholded_image = False 
show_contour = False
show_smooth_contour = False
show_linescan = False
show_bin_size_dependency = False

#MAIN PROGRAM:
#Collect all filenames
Data = []
# Search in the current directory (./) and subdirectories for .tif files
for root, dirs, files in os.walk("."):        
    for file in files:
            if file.endswith(file_format):
    # Add the Path of files to Data 
                    Data.append(os.path.join(root, file))
    

for filename in Data:
    if file_format == ".lsm":
        original = plt.imread(filename) # reads in .lsm as 
                                        # several greyscale images   
        memb_original = original[:,:,1]          
        actin_original = original[:,:, 0]        
    elif file_format == ".tif":
        for filename in Data:
            if filename.endswith("memb.tif"):
                memb_original = cv2.imread(filename,0)
            elif filename.endswith("actin.tif"):
                actin_original = cv2.imread(filename,0)

# To find the contour images are scaled to 255      
memb = IM.scale(memb_original)
x_contour, y_contour = IM.smoothed_contour(memb, cell_max_x, cell_max_y, 
                                           cell_min, threshold, pix_size, 
                                           linescan_length, n, 
                                           show_thresholded_image, 
                                           show_contour, show_smooth_contour)
# Linescans need to be done for the original, unscaled images
memb_ls, actin_ls, r= IM.smooth_Linescan(memb_original, actin_original,
                                      x_contour, y_contour, linescan_length)
rad = r * pix_size      # Radius in um
if show_linescan:
    plt.figure(4)
    plt.imshow(actin_ls, origin='lower',cmap = 'gray', 
               interpolation = 'bilinear',vmin=0,vmax=255)

# Calculation of Actin Cortex thickness:
# Obtain average Linescans:
average_memb = IM.average_Linescan(memb_ls, bin_size)
average_actin = IM.average_Linescan(actin_ls, bin_size)
average_rad = IM.average_Linescan(rad,bin_size)

# Obtain Membrane and Actin peaks via gauss fitting
memb_amp, memb_mean, memb_sigma = IM.fitten(average_memb, average_rad)
actin_amp, actin_mean, actin_sigma = IM.fitten(average_actin, average_rad)
# Obtain intra- and extracellular Intensity by averaging 10 points with 
# calc_dist distance from the peak in- and outside
Intensity_in, Intensity_out = IM.get_Intensities(average_actin, average_rad, 
                                                 actin_mean,calc_dist)
# Fitting is done by least square fitting in lmfit
h = []
i_c = []
# Write output file:
docname = filename.rstrip(file_format).rstrip('_memb')
doc = open(docname, 'w')
doc.write(docname + '\n Actin Cortex thickness measurement \n'+'bin_size = ' 
          + str(bin_size) + '\nObjective STD = '
          + str(sigma) + ' um' + ' \n \n' )

doc.write(('{:10} \t {:10} \t {:10} \t {:10} \t {:10} \t {:10} \t {:10} \t'
          + ' {:10} \t \n').format('Linescan', 'Memb_pos', 'Actin_pos', 
                               'Actin_height', 'Intensity_in', 'Intensity_out',
                               'i_c', 'h'))
for i in xrange(len(average_memb)):
    # Define parameters
    pars = TM.get_parameters_default(memb_mean[i], sigma, Intensity_in[i], 
                                     Intensity_out[i])
    # Perform the fit
    fit = lmfit.minimize(TM.residual, pars, kws = {'x_c': actin_mean[i],
                                                    'i_p': actin_amp[i]})
    i_c.append(fit.params["i_c"].value)
    h.append(fit.params["h"].value)
    doc.write(('{:10} \t {:10} \t {:10} \t {:10} \t {:10} \t {:10} \t {:10} \t'
              + ' {:10}  \n').format(str(i), memb_mean[i], actin_mean[i],
                                       actin_amp[i], Intensity_in[i], 
                                       Intensity_out[i], i_c[i], h[i] ))
    
doc.close()
# Calculate the average thickness over all Linescans
result = open(docname + '_results', 'w')
h_mean = np.mean(h)
h_std = np.std(h)
i_c_mean = np.mean(i_c)
i_c_std = np.std(i_c)
i_in_mean = np.mean(Intensity_in)
i_out_mean = np.mean(Intensity_out)

result.write(docname + '\nActin Cortex thickness measurement results after'
          + ' averaging over all bins \n' + 'bin_size = ' 
          + str(bin_size) + '\nObjective STD = '
          + str(sigma) + ' um' + ' \n \n' )

result.write(('{:10} \t {:10} \t {:10} \t {:10} \t {:10} \t  {:10} \n').format(
                                'Intensity_in', 'Intensity_out', 
                               'i_c', 'i_c_std', 'h',  'h_std'))
result.write(('{:10} \t {:10f} \t {:10f} \t {:10f} \t {:10f} \t {:10f} \n')
             .format(i_in_mean, i_out_mean, i_c_mean, i_c_std, h_mean, h_std ))
result.close()

print ('Calculating successful: Average cortex thickness is: {} um '
      .format(h_mean))
# Show bin_size dependency:
if show_bin_size_dependency:
    # Define array of Bin_sizes
    bin_size = range(10, n/2, 10)
    av_h = []
    h_list =[]
    i_c = []
    plt.figure(5, figsize =(16,16))
    for i in bin_size:
        average_memb = IM.average_Linescan(memb_ls, i, )
        average_actin = IM.average_Linescan(actin_ls, i, )
        average_rad = IM.average_Linescan(rad, i,)
        #Get intra- and extracellular intensities
        memb_amp, memb_mean, memb_sigma = IM.fitten(average_memb, average_rad)
        actin_amp, actin_mean, actin_sigma = IM.fitten(average_actin,
                                                       average_rad)
        Intensity_in, Intensity_out = IM.get_Intensities(average_actin,
                                                         average_rad, 
                                                         actin_mean,calc_dist)  
        thick = []
        for j in range(len(average_memb)):
            x_m = memb_mean[j]
            i_in, i_out = Intensity_in[j], Intensity_out[j]
            pars2 = TM.get_parameters_default(x_m, sigma, i_in, i_out)
            fit2 = lmfit.minimize(TM.residual, pars2, 
                                  kws={'x_c': actin_mean[j], 
                                       'i_p': actin_amp[j]})
            thick.append(fit2.params["h"].value)   
            i_c.append(fit2.params["i_c"].value)
        av_h.append(np.mean(thick))
        h_list.append(thick)
        plt.plot(i*np.ones(len(thick)),thick, c="b",ls="",marker = "+",ms = 10)
        
    plt.title("Bin Size dependency of delta")
    plt.plot(bin_size, av_h, 'o-', c = "r", label = "Mean Average Thickness")
    plt.xlabel('Bin Size', fontsize= 22)
    plt.ylabel('h [um]', fontsize= 22)
    plt.legend()
    plt.savefig(filename.rstrip(file_format)+"_"+str(np.max(bin_size)/len(bin_size))
                +'_binsteps.png')

plt.show()