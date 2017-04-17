""" 
Image Processing
https://github.com/saschahofmann/ActinCortex
@author: Sascha Hofmann, Biotec TU Dresden, Guck Lab
Following the paper 'Monitoring Actin Cortex Thickness in Live Cells' by 
Andrew G. Clark, Kai Dierkes and Ewa K. Paluch
Created: 20 April 2017
ImageProcess_v1
This Program contains functions for the Program ActinCortex.
"""
# Import Modules:
import numpy as np
import cv2
import scipy.ndimage
from scipy import exp
from scipy.interpolate import RectBivariateSpline
from lmfit import Model
import matplotlib.pyplot as plt
from matplotlib.path import Path
from pyefd import elliptic_fourier_descriptors

def scale(image):
    """Scales image to an 8-bit image increases the contrast
     and removes an offset
     """
    image = image - np.min(image) # remove minimum
    scaleup = 255.0/np.max(image) # scale to 255
    image = scaleup*image
    return image.astype(np.uint8) # bring the numbers in an integer 
                                  # format which opencv likes

def findContour(memb, cell_max_x, cell_max_y, cell_min, pix_size,
                show_threshold):
    """Finds the cell contour by thresholding and cv2.findcontours.
    Parameters
    ----------
    memb:    np.ndarray
             Image matrix in uint8 format
    cell_max_x/cell_max_y:     float
                               Maximum cell size in x and y direction in um.
    cell_min: float
              Minimum cell size in um
    pix_size: float
              Size of a single Pixel in um/pix 
              
    Return
    ------
    Cells: list
           List of dictionaries containing information about the contour lines
            between maximum and minimal values 
           """
    # Thresholding
    _, cache = cv2.threshold(memb, 200, 255, cv2.THRESH_TOZERO_INV)
    _, cache2 =cv2.threshold(cache, 60,255,cv2.THRESH_BINARY)
    
    # Closing holes
    binaryops = 5
    kernsize = 2*int(binaryops/2)+1
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernsize,kernsize))
    closing = cv2.morphologyEx(cache2, cv2.MORPH_CLOSE, element )
    
    # Show thresholded and closed images
    if show_threshold:
        plt.figure(1)
        plt.imshow(closing,origin='lower',cmap = 'gray', 
                interpolation = 'bilinear',vmin=0,vmax=255)   
        plt.show()
    
    #Conotur finding!
    _, raw_list, _ = cv2.findContours(closing, cv2.RETR_LIST,
                                      cv2.CHAIN_APPROX_NONE)
    IMAGE_WIDTH = int(memb.shape[1])
    IMAGE_HEIGHT = int(memb.shape[0])
    Cells = []
    for i_cont in range(len(raw_list)):
        if len(raw_list[i_cont]) < 10:
            continue
        x,y,w,h = cv2.boundingRect(raw_list[i_cont]) #Fits a bounding box 
                                                     #around the cell
        if ( #h < cell_max_y/pix_size and        #Throw out objects that 
             #w  < cell_max_x/pix_size and       #are too small/too large for us
             min(h,w) > cell_min/pix_size and
             x >2 and y > 2 and
             x+w < IMAGE_WIDTH -2 and #the cell should be in the image and not
             y+h < IMAGE_HEIGHT -2    #reach over the border of the image
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
            cell['contour'] = (raw_list[i_cont]).reshape(
                               raw_list[i_cont].shape[0],
                               raw_list[i_cont].shape[2])
            cell['raw_contour'] = (raw_list[i_cont])
            cell['area_pix'] =  mu['m00']
            Cells.append(cell)
    return Cells

def cart2pol(x, y):
    """Transforms coordinates from cartesian to polar."""
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
def pol2cart(rho, phi):
    """Transforms coordinates from polar to cartesian."""
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y) 
    
def index_coords(data, origin=None):
    """Creates x & y coords for the indicies in a numpy array "data".
    "origin" defaults to the center of the image. Specify origin=(0,0)
    to set the origin to the lower left corner of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_x, origin_y = origin
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x -= origin_x
    y -= origin_y
    return x, y

def reproject_image_into_polar(data, origin=None):
    """Reprojects a 3D numpy array ("data") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin = (nx//2, ny//2)

    # Determine that the min and max r and theta coords will be...
    x, y = index_coords(data, origin=origin)
    r, theta = cart2pol(x, y)

    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i = np.linspace(r.min(), r.max(), nx)
    theta_i = np.linspace(theta.min(), theta.max(), ny)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    xi, yi = pol2cart(r_grid, theta_grid)
    xi += origin[0] # We need to shift the origin back to 
    yi += origin[1] # back to the lower-left corner...
    xi, yi = xi.flatten(), yi.flatten()
    coords = np.vstack((xi, yi)) # (map_coordinates requires a 2xn array)
    # Reproject each band individually and the restack
    # (uses less memory than reprojection the 3-dimensional array in one step)
    # coords starts in the center of the contour map_coordinates 
    # takes the intensity value but puts it as first value into zi,
    # afterwards coords moves in circles and putes the intensity values into zi
    zi = scipy.ndimage.map_coordinates(data.T, coords, order=1)
    bands = (zi.reshape((nx, ny)))
    output = np.dstack(bands)
    return output, r_i, theta_i

def gauss(x,a,x0,sigma):
    """A 2D- gauss function with:
    Parameters:
    -----------
    x:  np.ndarray
        x-values of the function
    a:  float
        Height/Amplitude of the gauss function
    x0: float
        Mean of the gauss function
    sigma: float
        Standard deviation of the gauss function
        
    """
    return a*exp(-(x-x0)**2/(2*sigma**2))

def fitten(linescan, r):
    """Fits a gauss function around the maximum (5 points) of every individual
    linescan.The returned gauss array contains
    Output:
    -------
     amp:   list of floats
            the peak height 
     mean:  list of floats
            the mean value
     sigma: list of floats
            the standard deviation
     """
    # In case of a single Linescan the for-loop needs an additional Listwrap:
    if type(linescan[0]) == np.float64:
        linescan = [linescan]
    if type(r[0]) == np.float64:
        r = [r]
        
    amp, mean, sigma = [], [],[]
    for i in range(len(linescan)):
        # Make a gauss fit
        
        # Find maximum in the middle of the Linescan
        max_lin = np.argmax(linescan[i][30:70]) + 30
        # Take two points right and left of the maximum
        line_for_fitting = linescan[i][max_lin -2:max_lin + 3].astype(float)
        y = np.array(line_for_fitting) 
        # Create lmfit-Gauss-Model
        gmod = Model(gauss)
        # Define lmfit-Parameters
        x = r[i][max_lin -2:max_lin + 3]
        par = gmod.make_params( a = linescan[i][max_lin], 
                           x0 = r[i][max_lin], sigma = 0.2)
        par['x0'].set(min = x[0], max = x[-1])
        par['a'].set(min = 0)
        # Perform fitting
        results = gmod.fit(y, par, x=x)
        amp.append(results.params["a"].value)
        mean.append(results.params["x0"].value)
        sigma.append(results.params["sigma"].value)
    return amp, mean, sigma
                   
def first_contour(memb, cell_max_x, cell_max_y, cell_min, pix_size, 
                linescan_length, show_threshold, show_contour, 
                use_maxima = False):
    """Creates a contour on the maxima of the membrane by perfoming a 
    transformation to polar coordinates, linescanning (and optional choosing 
    the maxima as contour.
    Parameters:
    -----------
    memb:            np.ndarray
                     8bit image of the membrane
    cell_max_x/_y:   float or integer
                     Maximal size of a cell in x- and y-Dimension in um
    cell_min:        float or integer
                     Minimal size of a cell in um
    pix_size:        float
                     Size of a single pixel in um/pix
    linescan_length: integer
                     Length of the Linescan in pix
    show_threshold/
    show_contour:    Booleans
                     Toogle True to see the according Plot
    use_maxima:      Bool
                     Toogle True to use the maxima of the Linescans as 
                     line of contour
    """
    # Get the contour from cv.findcontour
    Cells = findContour(memb, cell_max_x, cell_max_y, cell_min, pix_size,
                        show_threshold)
    #Use the outer contour (larger area) 
    #to do a linescan, starting at the contour
    ind = np.argmax([Cells[i]['area_um'] for i in range(len(Cells))])

    x_contour,y_contour = Cells[ind]['contour'][:,0],Cells[ind]['contour'][:,1]
    pos_x = int(round(Cells[ind]['pos_x']))
    pos_y = int(round(Cells[ind]['pos_y']))
    x = x_contour
    y = y_contour
    # For an image with the highest intensity on the membrane only one can use
    # the maxima as a contour
    if use_maxima:
        #1.) Move the contour to the origin (0,0) by removing the centroid
        x_contour = x_contour - pos_x
        y_contour = y_contour - pos_y
        #2.) Translate contour to polar coordinates
        rho_contour,phi_contour  = cart2pol(x_contour,y_contour)
        #3.) Translate the picture to polar coordinates and define the pos_x and
        # pos_y as the center of the image
        img_polar_all = reproject_image_into_polar(memb, (pos_x,pos_y))
    
        # Create linescans ending at the outer Contour 
        # with a equal length : linescan_length
        # Make a gauss fit around the maximum
        # for each contour point find points in the picture with a similar angle 
        # (difference as close as zero as possible)
        
        Smallest_difference_rho, Smallest_difference_phi = [],[]
        phi = []
        radius = []
        for i in range(len(rho_contour)):
            smallest_difference_phi = np.argmin(abs(img_polar_all[2] 
                                                    - phi_contour[i])) 
            #TODO: find option without adding 30 px !
            smallest_difference_rho = np.argmin(abs(img_polar_all[1] 
                                                    - rho_contour[i])) +30 
            if smallest_difference_phi not in Smallest_difference_phi:
                
                #go towards inside the cell and get an intensity profile
                Smallest_difference_rho.append(smallest_difference_rho)  
                Smallest_difference_phi.append(smallest_difference_phi)
                # Take membran maximum as fix point
                linescan = img_polar_all[0][0][smallest_difference_phi,
                                              0:smallest_difference_rho]
                memb_max = np.argmax(linescan)
                # Write List of radius
                phi.append(phi_contour[i])
                radius.append(img_polar_all[1][memb_max])
        x, y = pol2cart(radius, phi)
        x = x + pos_x
        y = y + pos_y
    if show_contour:
        plt.figure(2)
        plt.imshow(memb, origin='lower',cmap = 'gray', 
                   interpolation = 'bilinear',vmin=0,vmax=255)
        plt.plot(x,y, c = 'r', linewidth = '2')   
    # Write contour in format for Fourier decomposition 
    contour = np.vstack((x,y)).T
    return contour, (pos_x, pos_y)

def smoothed_contour(memb, cell_max_x, cell_max_y, cell_min, pix_size,
                      linescan_length, n = 1000,show_threshold = False,
                      show_contour = False, show_smooth_con = False):
    """
    Creates a smooth contour through Fourier decomposition.
    Parameters:
    -----------
    memb:              np.ndarray
                       8bit image of the membrane
    cell_max_x/_y:     float or integer
                       Maximal size of a cell in x- and y-Dimension in um
    cell_min:          float or integer
                       Minimal size of a cell in um
    pix_size:          float
                       Size of a single pixel in um/pix
    linescan_length:   integer
                       Length of the Linescan in pix
    n:                 integer
                       Number of contour points calculated in the 
                       Fourier decomposition
    show_threshold,
    show_contour,
    show_smooth_con:   Booleans
                       Toogle True to see the according Plot

    """
    # Get raw contour
    contour, locus = first_contour(memb, cell_max_x, cell_max_y, cell_min,
                                   pix_size, linescan_length, show_threshold,
                                    show_contour )
    # Calculate Fourier coefficients up to order 10
    coeffs = elliptic_fourier_descriptors(contour, order=10, normalize= False)
    # Calculate the new contour coordinates
    x, y = fourierContour(coeffs, locus, n )
    if show_smooth_con:
        plt.figure(3)
        plt.imshow(memb, origin='lower',cmap = 'gray', 
                   interpolation = 'bilinear',vmin=0,vmax=255)
        plt.plot(x,y, c = 'r', linewidth = '2')
    return x, y

def smooth_Linescan(memb, actin, x_contour, y_contour, linescan_length):
    """
    Calculates a linescan from the original Image. A Linescan is the Intensity
    profile along a certain line.
    Using a smooth contour, orthogonal lines for each point of this contour with
    a length of linescan_length are calculated. The distance between two points
    on these lines is 1 pixel. The linescan values on these lines are 
    interpolated from the original images.
    
    Parameters:
    -----------
    memb/actin :         np.ndarray
                         8bit images of the membrane/actin
    x_contour/y_contour: np.ndarray
                         Coordinates of the contour
    linescan_length:     integer
                         Length of the Linescan in pix
                     
    Output:
    -------
    memb_ls/actin_ls:     np.ndarray
                          Linescans along orthogonal lines from the contour of 
                          actin and membrane images
    r:                    np.ndarray
                          Radius values for each Linescan in pixel
    """
    path = Path(np.vstack((x_contour,y_contour)).T)
    # Calculating the slope of our Linescan 
    # First calculate the tangent of the contour
    # The derivative at one point is the difference between 
    # This point and the next
    dx = np.diff(x_contour)
    dy = np.diff(y_contour)
    # Adding derivative for the last Point 
    dx = np.append(dx, x_contour[0]-x_contour[-1])
    dy = np.append(dy,y_contour[0]-y_contour[-1])
    derivative = dy/dx
    # The slope and the offset of the normal
    m = -1.0/derivative
    n = y_contour - m * x_contour
    # Calculate a matrix with the x values for every linescan
    # We want to have Linescans with linescan_length
    # Length in x-direction is depending on the slope
    x_length = -m/(m**2+1) + np.sqrt((m/(m**2+1))**2 
                                     +linescan_length**2/(1+m**2))
    x = np.tile(x_contour[np.newaxis].T, (1, linescan_length))
    x_steps = np.tensordot(np.arange(-linescan_length/2, linescan_length/2, 1),
                           x_length.T/linescan_length, axes = 0).T
    x = x + x_steps
    # Calculate linear function for Linescan
    y = m[np.newaxis].T * x + n[np.newaxis].T
    # Create the interpolations for membrane and cortex
    memb_interpol = RectBivariateSpline(np.arange(memb.shape[0]), 
                                        np.arange(memb.shape[1]), memb)
    actin_interpol = RectBivariateSpline(np.arange(actin.shape[0]), 
                                         np.arange(actin.shape[1]), actin)
    memb_ls = memb_interpol.ev(y, x)
    actin_ls = actin_interpol.ev(y, x)
    # Make sure Linescans go from inside to outside
    mask = path.contains_points(np.vstack((x[:,0], y[:,0])).T)
    mask = np.invert(mask)
    memb_ls[mask, :] = memb_ls[mask, :][:,::-1]
    actin_ls[mask, :]= actin_ls[mask, :][:,::-1]
    r = np.sqrt((x - (np.ones(x.shape).T*x[:,0]).T)**2 
                + (y - (np.ones(y.shape).T*y[:,0]).T)**2)
    return memb_ls, actin_ls, r
    
    
    
def average_Linescan(Linescan, bin_size, overlap = 0):
    """Sums up individual Linescans and averages them. The number of averaged 
    Linescans is determined by bin_size.
    
    Parameters:
    -----------
    linescan: np.ndarray
              Array of Linescans
    bin_size: integer
              Number of Linescans over which the averaging is performed
    overlap:  integer
              optional parameter, number of linescans which bins overlap
    """
    average_Linescan = []
    a = 0
    # As the number of linescans does not necessarily have to be a multiple of 
    # the bin_size the last averaging needs to eventually take some of the first
    # linescans into account again
    while a < len(Linescan):
        if a + bin_size <= len(Linescan):
            average_Linescan.append(np.mean(Linescan[a: a + bin_size], 0))
        else:
            cut = np.concatenate((Linescan[a: len(Linescan)],
                                  Linescan[:bin_size - len(Linescan) + a]),0)
            average_Linescan.append(np.mean(cut, 0))
        a += bin_size - overlap
    return average_Linescan

def get_Intensities(average_actin, rad, mean, calc_dist):
    """
    Calculates the intra- and extracellular Intensities from a linescan.
    The function finds the points with the distance 'calc_cist' from the peak
    in- and inside of the cell. It then averages over the next 10 points of the
    linescan.
    
    Parameters:
    -----------
    average_actin:    np.ndarray
                      Array of linescans from an actin stained image
    rad:              np.ndarray
                      Array of radius 
    mean:             list of floats
                      Peak location in the linescan
    calc_dist:        float
                      Distance from the peak at which the Intensities are 
                      calculated, in um
    """
    # For single linescans an additional list layer needs to be added for the 
    # for loop
    if type(average_actin[0])== np.float64:
        average_actin = [average_actin]
    if type(rad[0]) == np.float64:
        rad = [rad]
    Intensity_in, Intensity_out = [],[]
    for i in range(len(average_actin)):
        # Find argument with distance calc_dist from the peak in- and 
        # outside of the cell
        r_in = mean[i]- calc_dist
        r_out = mean[i] + calc_dist 
        r_in = np.argmin(abs(r_in - rad[i])) 
        r_out = np.argmin(abs(r_out - rad[i]))
        # Mean of the 10 next pixel to those distances
        i_in = np.mean(average_actin[i][r_in-10: r_in ]) 
        i_out = np.mean(average_actin[i][r_out: r_out +10])
        Intensity_in.append(i_in)
        Intensity_out.append(i_out)
    return Intensity_in, Intensity_out

def fourierContour(coeffs, locus=(0., 0.), n=300):
    """Plot a ``[2 x (N / 2)]`` grid of successive truncations of the series.
    .. note::
        Requires `matplotlib <http://matplotlib.org/>`_!
    :param numpy.ndarray coeffs: ``[N x 4]`` Fourier coefficient array.
    :param list, tuple or numpy.ndarray locus:
        The :math:`A_0` and :math:`C_0` elliptic locus in [#a]_ and [#b]_.
    :param int n: Number of points to use for plotting of Fourier series.
    """
    try:
        _range = xrange
    except NameError:
        _range = range
    t = np.linspace(0, 1.0, n, endpoint = False)
    xt = np.ones((n,)) * locus[0]
    yt = np.ones((n,)) * locus[1]
    for n in _range(coeffs.shape[0]):
        xt += (coeffs[n, 0] * np.cos(2 * (n + 1) * np.pi * t)) + \
              (coeffs[n, 1] * np.sin(2 * (n + 1) * np.pi * t))
        yt += (coeffs[n, 2] * np.cos(2 * (n + 1) * np.pi * t)) + \
              (coeffs[n, 3] * np.sin(2 * (n + 1) * np.pi * t))
    return xt, yt
