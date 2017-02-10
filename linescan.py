#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.ticker import NullFormatter
import pandas as pd
import scipy.ndimage
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import ImageProcess as f

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

# Define Pixelsize, Cellmax and min sizen, bin size for averaging and the length of the linescans
#TODO: include automated pixelsize recognition
pix_size = 64.02/512  #um/pix
cell_max_y = 25
cell_max_x = 25
cell_min = 10
linescan_length = 100
#bin_size =range(1, 251,1)
bin_size = 100
corr = 1.0015       # Magnification correction factor for chromatic shift
calc_dist = 1       # Dist from peak for i_in and i_out calculation in um
#Panda.DataFrame is tabular-like with column and row labelling possible ??
df_ALL = pd.DataFrame()    #?????

for filename in Data:
    # img = cv2.imread(filename,1) # putting a 1 there tells him to give us all the three channels (RGB)
                                    # yellow should be composed of a bit red and a bit green 
                                    # but for simplicity we lookonly in one channel:
    
    original = plt.imread(filename) # reads in .tif and .lsm as several greyscale images
    memb = original[:,:,1]           # membrane picture should be pic number 1
    actin = original[:,:, 0]        # actin cortex number 0
    memb = f.scale(memb)
    actin = f.scale(actin)
    _, cache = cv2.threshold(memb, 200, 255, cv2.THRESH_TOZERO_INV)
    _, cache2 =cv2.threshold(cache, 60,255,cv2.THRESH_BINARY)
    #plt.figure(1)
    #plt.imshow(cache2,origin='lower',cmap = 'gray', interpolation = 'bilinear',vmin=0,vmax=255)
    
    binaryops = 5
    kernsize = 2*int(binaryops/2)+1
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernsize,kernsize))
    closing = cv2.morphologyEx(cache2, cv2.MORPH_CLOSE, element )
    #closing = cv2.dilate(closing, 2*element,5)
    #cache = cv2.dilate(cache2, element)
    #cache2= cv2.erode(cache, element)
    #plt.figure(1)
    #plt.imshow(closing,origin='lower',cmap = 'gray', interpolation = 'bilinear',vmin=0,vmax=255)   
    #plt.show()
#    Conotur finding!
    _, raw_list, _ = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    IMAGE_WIDTH = int(memb.shape[1])
    IMAGE_HEIGHT = int(memb.shape[0])
    Cells = []
    for i_cont in range(len(raw_list)):
        # ???
        if len(raw_list[i_cont]) < 10:
            continue
        x,y,w,h = cv2.boundingRect(raw_list[i_cont]) #Fits a bounding box around the cell
        if ( h < cell_max_y/pix_size and  #through out objects that are too small/too large for us
             w  < cell_max_x/pix_size and
             min(h,w) > cell_min/pix_size and
             x >2 and y > 2 and
             x+w < IMAGE_WIDTH -2 and #the cell should be in the image and not reach over the border of the image
             y+h < IMAGE_HEIGHT -2
             ):
            arc = cv2.arcLength(raw_list[i_cont], True)
            #area= cv2.contourArea(raw_list[i_cont])
            mu = cv2.moments(raw_list[i_cont], False)
            cell = {}
            cell['height'] = h*pix_size
            cell['width'] = w*pix_size
            cell['pos_x'] = mu['m10']/mu['m00']
            cell['pos_y'] = mu['m01']/mu['m00']
            cell['area_um'] = mu['m00']*pix_size*pix_size
            cell['contour'] = (raw_list[i_cont]).reshape(raw_list[i_cont].shape[0],raw_list[i_cont].shape[2])
            cell['area_pix'] =  mu['m00']
            Cells.append(cell)
    """ 
    #Plotting of the contour
    nullfmt   = NullFormatter() 
    plt.figure(3,figsize=(8,8))
    plt.imshow(memb,cmap = 'gray', interpolation = 'bilinear',vmin=0,vmax=255)
    for i in range(len(Cells)):
        contour = Cells[i]['contour']
        plt.plot(contour[:,0],contour[:,1],label='Size='+str(Cells[i]['area_um']))
        plt.legend(loc='upper right', fancybox=True, framealpha=0.5)
    plt.axis('off')
    plt.savefig(filename+'_Contours.png')
    plt.show()
    plt.close()           
    #Use the outer contour (larger area) to do a linescan, starting at the conotur
    """
    ind = np.argmax([Cells[i]['area_um'] for i in range(len(Cells))])
    
    #1.) Move the contour to the origin (0,0) by removing the centroid
    x_contour,y_contour = Cells[ind]['contour'][:,0],Cells[ind]['contour'][:,1]
    pos_x = int(round(Cells[ind]['pos_x']))
    pos_y = int(round(Cells[ind]['pos_y']))
  
    x_contour = x_contour - pos_x
    y_contour = y_contour - pos_y
    #2.) Translate contour to polar coordinates
    rho_contour,phi_contour  = f.cart2pol(x_contour,y_contour)     

    #3.) Translate the picture to polar coordinates and define the pos_x and pos_y as the center of the image
    img_polar_all = f.reproject_image_into_polar(memb, (pos_x,pos_y))
    img_polar = img_polar_all[0].reshape(memb.shape)
    actin_polar_all = f.reproject_image_into_polar(actin, (pos_x,pos_y))
    actin_polar = actin_polar_all[0].reshape(actin.shape)
    #plt.figure(3)
    #plt.imshow(actin_polar_all[0][0],origin='lower',cmap = 'gray', interpolation = 'bilinear',vmin=0,vmax=255)
    #plt.show()
    
    # Create linescans ending at the outer Contour with a equal length : linescan_length
    # Make a gauss fit around the maximum
    # for each contour point find points in the picture with a similar angle (difference as close as zero as possible)
    
    Smallest_difference_rho, Smallest_difference_phi = [],[]
    memb_ls, actin_ls = [],[] 
    memb_gauss, actin_gauss = [],[]
    radius, delta = [], []
    for i in range(len(rho_contour)):
        smallest_difference_phi = np.argmin(abs(img_polar_all[2] - phi_contour[i])) #argmin gives the index of the minimum
        #TODO: find option without adding 30 px !
        smallest_difference_rho = np.argmin(abs(img_polar_all[1] - rho_contour[i])) +30  #argmin gives the index of the minimum
        if smallest_difference_phi not in Smallest_difference_phi:
            #go towards inside the cell and get an intensity profile
            Smallest_difference_rho.append(smallest_difference_rho)  
            Smallest_difference_phi.append(smallest_difference_phi)
            # Take Contour as fix point make Linescans with equal length
            linescan = img_polar_all[0][0][smallest_difference_phi,smallest_difference_rho - linescan_length:smallest_difference_rho]
            memb_ls.append(linescan)
            linescan2 = actin_polar_all[0][0][smallest_difference_phi,smallest_difference_rho - linescan_length:smallest_difference_rho]
            actin_ls.append(linescan2)
            # Write List of radius
            r = img_polar_all[1][smallest_difference_rho - linescan_length:smallest_difference_rho]
            radius.append(r)
            #Make Gauss Fit
            popt = f.fitten(linescan,r)          # Membranfit
            memb_gauss.append(popt)
            
            aopt = f.fitten(linescan2,r)        # Actinfit
            actin_gauss.append(aopt)
            
            # Calculate delta
            delta.append(popt[1] - aopt[1])
    
    single_delta = np.mean(delta)      
    """
    plt.figure(2,figsize =(8,16))
    for i in range(6):
          
        a, mean, sigma = memb_gauss[i][0] , memb_gauss[i][1], memb_gauss[i][2]        
        plt.subplot(3,2,i+1)  
        plt.plot(radius[i],memb_ls[i],'b+:',label='data')
        plt.plot(radius[i],f.gauss(radius[i],a ,mean,sigma),'r-',label='fit') 
        plt.plot(radius[i],actin_ls[i],"g+:", label ="actin")
        a, mean, sigma = actin_gauss[i][0] , actin_gauss[i][1], actin_gauss[i][2]
        plt.plot(radius[i],f.gauss(radius[i],a ,mean,sigma),'y-',label='fit')
        plt.title(i)
    plt.show()
    
    print np.mean(delta)
     
    
    bin_size = range(1, len(memb_ls)+1,1)
    Av_delta = []
    for i in bin_size:
        average_memb = f.average_Linescan(memb_ls, i, i/4)
        average_actin = f.average_Linescan(actin_ls, i, i /4)
        rad = f.average_Linescan(radius, i, i/4)
        average_delta = []
        for j in range(len(average_memb)):
            # Find maximum of Linescan
            max_lin = np.argmax(average_memb[j])
            # Get Gauss-parameter
            popt = f.fitten(average_memb[j],rad[j])
            memb_mean = popt[1]
            # Find maximum of Linescan
            max_lin = np.argmax(average_actin[j])
        
            # Get Gauss-parameter
            aopt = f.fitten(average_actin[j], rad[j])
            actin_mean = aopt[1]
            
            average_delta.append(memb_mean- actin_mean)
        Av_delta.append(np.mean(average_delta))    
    plt.figure(1, figsize =(16,16))
    plt.title("Bin Size dependency of delta")
    plt.plot(bin_size, Av_delta, 'o-', label = "Bin Size dependency")
    plt.axhline(single_delta,  c = "r", label = "Delta calculated from single Linescans")
    plt.xlabel('Bin Size')
    plt.ylabel('Delta [pix]')
    plt.legend()
    plt.savefig(filename.rstrip(".lsm")+"_"+str(np.max(bin_size)/len(bin_size))+'_binsteps_0.25_overlap.png')
    plt.show()
    """
  
    # Average bin_size linescans
    average_memb = f.average_Linescan(memb_ls, bin_size)
    average_actin = f.average_Linescan(actin_ls, bin_size)
    
    #Calculate Constants i_in and i_out
           
    average_delta= [] 
    Intensity_in, Intensity_out = [],[]                       
    rad = f.average_Linescan(radius, bin_size)
    memb_mean, actin_mean = [],[]
    plt.figure(1, figsize=(16,16))
    for i in range(len(average_memb)):
        # Find maximum of Linescan
        max_lin = np.argmax(average_memb[i])
        # Get Gauss-parameter
        popt = f.fitten(average_memb[i],rad[i])
        memb_mean.append(popt[1])
        # Find maximum of Linescan
        max_lin = np.argmax(average_actin[i])
    
        # Get Gauss-parameter
        aopt = f.fitten(average_actin[i], rad[i])
        actin_mean.append(aopt[1])
        
        average_delta.append(popt[1]- aopt[1])
         
        # calculate i_in and i_out
        # find argument with distance calc_dist from the peak in- and outside of the cell
        r_in = aopt[1]- calc_dist/pix_size
        r_out = aopt[1] + calc_dist/pix_size 
        r_in = np.argmin(abs(r_in - rad[i])) 
        r_out = np.argmin(abs(r_out - rad[i]))
        
        # Mean of the 10 next pixel to those distances
        i_in = np.mean(average_actin[i][r_in-10: r_in ]) 
        i_out = np.mean(average_actin[i][r_out: r_out +10])
        Intensity_in.append(i_in)
        Intensity_out.append(i_out)
       
        
        # Plot
        plt.subplot(3,2,i+1)
        plt.title(i)
        #Membranlinescanplot
        plt.plot(rad[i],average_memb[i],ls = '', marker = '.')
        gauss = f.gauss(rad[i],popt[0], popt[1], popt[2])
        plt.plot(rad[i],gauss)
        # Actinlinescanplot
        plt.plot(rad[i],average_actin[i],ls = '', marker = '.')
        gauss2 = f.gauss(rad[i],aopt[0], aopt[1], aopt[2])#memb_gauss[i][0], memb_gauss[i][1], memb_gauss[i][2])
        plt.plot(rad[i],gauss2)
        # Intensityconstantsplot 
        plt.plot(rad[i][r_in-10:r_in], average_actin[i][r_in-10:r_in], c = "cyan", marker = "o")
        plt.plot(rad[i][r_out:r_out+10], average_actin[i][r_out:r_out+10], c = "cyan", marker = "o")
        plt.xlabel('Rho [pix]')
        plt.ylabel('Linescan (Intensity)')
    plt.savefig(filename.rstrip(".lsm")+'_average_Linescan.png')
    plt.show()   
    plt.close()    
    
 
    """   
    #plot these point in the polar image
    plt.imshow(img_polar,origin='lower',cmap = 'gray', interpolation = 'bilinear',vmin=0,vmax=255)
    plt.scatter(Smallest_difference_rho,Smallest_difference_phi,s=2,c='red',marker='o',linewidth='0')
    plt.xlabel('Rho')
    plt.ylabel('Phi')
    plt.show()
    """