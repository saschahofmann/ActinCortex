# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 16:33:36 2017

@author: shadaabuhattum
"""

import scipy.integrate as integrate
import lmfit 
import numpy as np


def gauss(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))  
  
def cdf(x,sigma):
    """Calculates the cumulative distribution function for a gauss distributed probability density"""
    if type(x) is np.float64:
        inte = (integrate.quad(gauss, - np.inf, x, 
                                args=(1/np.sqrt(2* np.pi * sigma**2),0,sigma)))[0]
    else:
        inte = np.zeros_like(x)
        for i in range(len(x)):
            inte[i]=(integrate.quad(gauss, - np.inf, x[i], 
                                    args=(1/np.sqrt(2* np.pi * sigma**2),0,sigma)))[0]
    return inte

def convolution(x,h, i_c, x_m, sigma, i_in, i_out):
    """Theoretical Model for a Actin Cortex with a thickness below the diffraction limit. 
    Assumptions:
    ------------
    Intra-, extra- and actin cortex intensity are assumed to be constant, the rectangle function is convolved with 
    a gauss distribution with mean = 0 and standard deviation = sigma  to mimick diffraction.
         
    Parameters: 
    ----------
    x:  np.ndarray
            Distance from cell center in pixel
    i_c: float
             Intensity height of the actin cortex before convolving
    delta: float
               Distance between membran peak and actin peak in the measured data in pixel
    x_m: float
             Membrane peak distance from the cell center in pixel
    sigma: float
               Standard deviation of diffraction gauss
    i_in: float
              Intracellular intensity
    i_out: float
               Extracellular intensity
               
    Returns
        -------
    L: np.ndarray
           Actin distribution   
    """ 
    smalldelta =  sigma**2 / h * np.log((i_out -i_c)/(i_in - i_c))      
    delta =  h/2 + smalldelta       
    X_c = x_m - delta + smalldelta 
    Conv = i_in + (i_c - i_in)*cdf(x - X_c + h/2, sigma) \
                + (i_out -i_c)*cdf(x - X_c - h/2, sigma)
    return Conv
    
    
def model(params, x):
    return convolution(x = x,
                       h = params["h"].value,
                       i_c = params["i_c"].value,
                       x_m = params["x_m"].value,
                       sigma = params["sigma"].value,
                       i_in = params["i_in"].value,
                       i_out = params["i_out"].value)
        
def residual(params, x, linescan):
    """Computes residuals (= difference between model and data points) for fitting
        
    Parameters
    ----------
    params: lmfit.Parameters
                The fitting parameters for `model`
     x:     np.ndarray
               Distance from cell center in pixel
    linescan: np.ndarray
                  Experimental data --> measured Intensity"""
    return linescan - model(params, x)


def get_parameters_default(delta, x_m,sigma, i_in, i_out):
    # The order of the parameters must match the order 
    # of 'parameter_names' and 'parameter_keys'.
    params = lmfit.Parameters()
    params.add("h", value =0.1 , min = 0.01, max =1.)
    params.add("i_c", value = 200, min = i_in+1., max = 500)
    params.add("x_m", value = x_m, vary = False)
    params.add("sigma", value = sigma, vary = False)
    params.add("i_in", value = i_in, vary = False)
    params.add("i_out", value = i_out, vary = False)
    return params


def checker(h, i_c, x_m, sigma, i_in, i_out):
    smalldelta = sigma**2/h*np.log((i_out - i_c)/(i_in - i_c))
    x_c = x_m - h/2 - smalldelta
    return x_c

def model2(params):
    return checker(h = params["h"].value,
                   i_c = params["i_c"].value,
                   x_m = params["x_m"].value,
                   sigma = params["sigma"].value,
                   i_in = params["i_in"].value,
                   i_out = params["i_out"].value )
    
def residual2(params, x_c, i_p):
    
    res1 = x_c - model2(params)
    res2 = i_p - model(params, model2(params))
    return res1, res2

def get_parameters_default2(x_m, sigma, i_in, i_out):
    # The order of the parameters must match the order 
    # of 'parameter_names' and 'parameter_keys'.
    params = lmfit.Parameters()
    params.add("h", value =0.1 , min = 0.0001, max =1.)
    params.add("i_c", value = 200., min = i_in+1., max = 1000)
    params.add("x_m", value = x_m, vary = False)
    params.add("sigma", value = sigma, vary = False)
    params.add("i_in", value = i_in, vary = False)
    params.add("i_out", value = i_out, vary = False)
    return params 

   