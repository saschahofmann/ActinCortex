import numpy as np
import cv2
import scipy.ndimage
from scipy import exp
from lmfit import Model
import matplotlib.pyplot as plt
from pyefd import elliptic_fourier_descriptors
from pyefd import plot_efd

def scale(image):
    """Scales image to an 8-bit image increases the contrast
     and removes an offset"""
    image = image - np.min(image) # remove minimum
    scaleup = 255.0/np.max(image) # scale to 255
    image = scaleup*image
    return image.astype(np.uint8) # bring the numbers in an integer 
                                  # format which opencv likes

def findContour(memb, cell_max_x, cell_max_y, cell_min, pix_size):
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
    _, cache = cv2.threshold(memb, 200, 255, cv2.THRESH_TOZERO_INV)
    _, cache2 =cv2.threshold(cache, 60,255,cv2.THRESH_BINARY)
    
    
    binaryops = 5
    kernsize = 2*int(binaryops/2)+1
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernsize,kernsize))
    closing = cv2.morphologyEx(cache2, cv2.MORPH_CLOSE, element )
    
    #plt.figure(1)
    #plt.imshow(closing,origin='lower',cmap = 'gray', 
    #           interpolation = 'bilinear',vmin=0,vmax=255)   
    #plt.show()
    
    #Conotur finding!
    _, raw_list, _ = cv2.findContours(closing, cv2.RETR_LIST,
                                      cv2.CHAIN_APPROX_NONE)
    IMAGE_WIDTH = int(memb.shape[1])
    IMAGE_HEIGHT = int(memb.shape[0])
    Cells = []
    for i_cont in range(len(raw_list)):
        # ???
        if len(raw_list[i_cont]) < 10:
            continue
        x,y,w,h = cv2.boundingRect(raw_list[i_cont]) #Fits a bounding box 
                                                     #around the cell
        if ( h < cell_max_y/pix_size and        #Throw out objects that 
             w  < cell_max_x/pix_size and       #are too small/too large for us
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
            cell['contour'] = (raw_list[i_cont]).reshape(raw_list[i_cont].shape[0],
                                                         raw_list[i_cont].shape[2])
            cell['raw_contour'] = (raw_list[i_cont])
            cell['area_pix'] =  mu['m00']
            Cells.append(cell)
    return Cells

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
def pol2cart(rho, phi):
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
    return a*exp(-(x-x0)**2/(2*sigma**2))

def fitten(linescan, r):
    """Afterwards a gauss fit around the maximum (5 points) of every individual
    linescan is approximated.The returned gauss array contains 0: the peak
    height, 1: the mean value and 2: the standard deviation"""
    amp, mean, sigma = [], [],[]
    for i in range(len(linescan)):
        # Make a gauss fit
        max_lin = np.argmax(linescan[i])
        # Choose two points right and left of the maximum
        line_for_fitting = linescan[i][max_lin -2:max_lin + 3].astype(float)
        y = np.array(line_for_fitting) 
        gmod = Model(gauss)
        x = r[i][max_lin -2:max_lin + 3]
        results = gmod.fit(y, x = x, a = linescan[i][max_lin], 
                           x0 = r[i][max_lin], sigma = 0.2)
        amp.append(results.params["a"].value)
        mean.append(results.params["x0"].value)
        sigma.append(results.params["sigma"].value)
    return amp, mean, sigma
        

def linescan(memb, actin, cell_max_x, cell_max_y, cell_min, pix_size, linescan_length):
    """Creates Linescans from polar images. Firstly image points close to the 
    contour line are searched these points are end points of the according 
    linescan. The linescans have an equal length --> linescan_length.
    """
    
    Cells = findContour(memb, cell_max_x, cell_max_y, cell_min, pix_size)
    #Use the outer contour (larger area) 
    #to do a linescan, starting at the conotur
    ind = np.argmax([Cells[i]['area_um'] for i in range(len(Cells))])
    
    #1.) Move the contour to the origin (0,0) by removing the centroid
    x_contour,y_contour = Cells[ind]['contour'][:,0],Cells[ind]['contour'][:,1]
    pos_x = int(round(Cells[ind]['pos_x']))
    pos_y = int(round(Cells[ind]['pos_y']))
    x_contour = x_contour - pos_x
    y_contour = y_contour - pos_y
    #2.) Translate contour to polar coordinates
    rho_contour,phi_contour  = cart2pol(x_contour,y_contour)   
    #3.) Translate the picture to polar coordinates and define the pos_x and
    # pos_y as the center of the image
    img_polar_all = reproject_image_into_polar(memb, (pos_x,pos_y))
    actin_polar_all = reproject_image_into_polar(actin, (pos_x,pos_y))
    
    # Create linescans ending at the outer Contour 
    # with a equal length : linescan_length
    # Make a gauss fit around the maximum
    # for each contour point find points in the picture with a similar angle 
    # (difference as close as zero as possible)
    
    Smallest_difference_rho, Smallest_difference_phi = [],[]
    memb_ls, actin_ls = [],[] 
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
            """
            # Take Contour as fix point make Linescans with equal length
            linescan = img_polar_all[0][0][smallest_difference_phi,
                                           smallest_difference_rho - 
                                           linescan_length:smallest_difference_rho]
            
            memb_ls.append(linescan)
            linescan2 = actin_polar_all[0][0][smallest_difference_phi,
                                              smallest_difference_rho 
                                              - linescan_length:smallest_difference_rho]
            actin_ls.append(linescan2)
            # Write List of radius in um
            r = img_polar_all[1][smallest_difference_rho - 
                                 linescan_length:smallest_difference_rho]
            radius.append(r*pix_size)
            """
            # Take membran maximum as fix point
            linescan = img_polar_all[0][0][smallest_difference_phi,
                                          0:smallest_difference_rho]
            #r = img_polar_all[1][0:smallest_difference_rho]
            #__, memb_max, __ = fitten([linescan],[r])
            #memb_max = np.argmin(abs(r - memb_max))
            memb_max = np.argmax(linescan)
            linescan = img_polar_all[0][0][smallest_difference_phi, memb_max - linescan_length/2: memb_max + linescan_length/2]
            
            memb_ls.append(linescan)
            linescan2 = actin_polar_all[0][0][smallest_difference_phi, memb_max - linescan_length/2: memb_max + linescan_length/2]
            actin_ls.append(linescan2)
            # Write List of radius in um
            r = img_polar_all[1][memb_max - linescan_length/2: memb_max + linescan_length/2]
            radius.append(r*pix_size)
            
    return memb_ls, actin_ls, radius  

           
def contour_1(memb, cell_max_x, cell_max_y, cell_min, pix_size, linescan_length):
    Cells = findContour(memb, cell_max_x, cell_max_y, cell_min, pix_size)
    #Use the outer contour (larger area) 
    #to do a linescan, starting at the conotur
    ind = np.argmax([Cells[i]['area_um'] for i in range(len(Cells))])
    
    #1.) Move the contour to the origin (0,0) by removing the centroid
    x_contour,y_contour = Cells[ind]['contour'][:,0],Cells[ind]['contour'][:,1]
    pos_x = int(round(Cells[ind]['pos_x']))
    pos_y = int(round(Cells[ind]['pos_y']))
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
    memb_ls = [] 
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
            #r = img_polar_all[1][0:smallest_difference_rho]
            #__, memb_max, __ = fitten([linescan],[r])
            #memb_max = np.argmin(abs(r - memb_max))
            memb_max = np.argmax(linescan)

            # Write List of radius in um
            phi.append(phi_contour[i])
            radius.append(img_polar_all[1][memb_max])
    x, y = pol2cart(radius, phi)
    x = x + pos_x
    y = y + pos_y
    contour = np.vstack((x,y)).T
    return contour, (pos_x, pos_y)

def contour_2(contour, locus):
    x_contour, y_contour = contour[0], contour[1]

    #contour = np.ndarray(x_contour, y_contour)
    coeffs = elliptic_fourier_descriptors(contour, order=10, normalize= False)
    x, y = fourierContour(coeffs, locus = locus, n = 5000)
    return (x, y)

def smooth_Linescan(memb, contour, linescan_length, pix_size):
    contour = np.vstack((contour[0],contour[1])).T
    pos_x, pos_y = centroid_of_polygon(contour)
    x_contour = contour[:,0] - pos_x
    y_contour = contour[:,1] - pos_y
    #2.) Translate contour to polar coordinates
    rho_contour,phi_contour  = cart2pol(x_contour,y_contour)
    #3.) Translate the picture to polar coordinates and define the pos_x and
    # pos_y as the center of the image
    #TODO: get rid of int()
    img_polar_all = reproject_image_into_polar(memb, (int(pos_x),int(pos_y)))
    # Create linescans ending at the outer Contour 
    # with a equal length : linescan_length
    # Make a gauss fit around the maximum
    # for each contour point find points in the picture with a similar angle 
    # (difference as close as zero as possible)
    
    Smallest_difference_rho, Smallest_difference_phi = [],[]
    memb_ls = [] 
    radius = []
    for i in range(len(rho_contour)):
        smallest_difference_phi = np.argmin(abs(img_polar_all[2] 
                                                - phi_contour[i])) 
        #TODO: find option without adding 30 px !
        smallest_difference_rho = np.argmin(abs(img_polar_all[1] 
                                                - rho_contour[i])) 
        if smallest_difference_phi not in Smallest_difference_phi:
            
            #go towards inside the cell and get an intensity profile
            Smallest_difference_rho.append(smallest_difference_rho)  
            Smallest_difference_phi.append(smallest_difference_phi)

            # Take membran maximum as fix point
            linescan = img_polar_all[0][0][smallest_difference_phi,
                                          smallest_difference_rho - linescan_length/2:
                                          smallest_difference_rho + linescan_length/2]
            r = img_polar_all[1][smallest_difference_rho - linescan_length/2:
                                          smallest_difference_rho + linescan_length/2]
            memb_ls.append(linescan)
            radius.append(r*pix_size)
    return memb_ls, radius     
    
    
    
def average_Linescan(Linescan, bin_size, overlap = 0):
    """Sums up individual Linescans and averages them. The number of averged 
    Linescans is determined by bin_size."""
    average_Linescan = []
    a = 0
    while a < len(Linescan):
        if a + bin_size <= len(Linescan):
            #average_Linescan.append(np.sum(Linescan[a: a + bin_size], 0)/float(bin_size))
            average_Linescan.append(np.mean(Linescan[a: a + bin_size], 0))
        else:
            cut = Linescan[a: len(Linescan)] + Linescan[:bin_size - len(Linescan) + a ]
            #average_Linescan.append(np.sum(cut,0)/float(bin_size))
            average_Linescan.append(np.mean(cut, 0))
        a += bin_size - overlap
    return average_Linescan

def get_Intensities(average_actin, rad, mean, calc_dist):
    Intensity_in, Intensity_out = [],[]
    for i in range(len(average_actin)):
        
        # Get Gauss-parameter
        # calculate i_in and i_out
        # find argument with distance calc_dist from the peak in- 
        # and outside of the cell
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
    t = np.linspace(0, 1.0, n)
    xt = np.ones((n,)) * locus[0]
    yt = np.ones((n,)) * locus[1]
    for n in _range(coeffs.shape[0]):
        xt += (coeffs[n, 0] * np.cos(2 * (n + 1) * np.pi * t)) + \
              (coeffs[n, 1] * np.sin(2 * (n + 1) * np.pi * t))
        yt += (coeffs[n, 2] * np.cos(2 * (n + 1) * np.pi * t)) + \
              (coeffs[n, 3] * np.sin(2 * (n + 1) * np.pi * t))
    return xt, yt

def area_of_polygon(x, y):
    """Calculates the signed area of an arbitrary polygon given its verticies
    http://stackoverflow.com/a/4682656/190597 (Joe Kington)
    http://softsurfer.com/Archive/algorithm_0101/algorithm_0101.htm#2D%20Polygons
    """
    area = 0.0
    for i in xrange(-1, len(x) - 1):
        area += x[i] * (y[i + 1] - y[i - 1])
    return abs(area) / 2.0

def centroid_of_polygon(points):
    """
    http://stackoverflow.com/a/14115494/190597 (mgamba)
    Centroid of polygon: http://en.wikipedia.org/wiki/Centroid#Centroid_of_polygon
    """
    import itertools as IT
    area = area_of_polygon(*zip(*points))
    result_x = 0
    result_y = 0
    N = len(points)
    points = IT.cycle(points)
    x1, y1 = next(points)
    for i in range(N):
        x0, y0 = x1, y1
        x1, y1 = next(points)
        cross = (x0 * y1) - (x1 * y0)
        result_x += (x0 + x1) * cross
        result_y += (y0 + y1) * cross
    result_x /= (area * 6.0)
    result_y /= (area * 6.0)
    return abs(result_x), abs(result_y)
                    