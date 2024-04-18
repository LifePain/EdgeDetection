import skimage
import scipy
import numpy as np
from skimage import feature

def canny_edge_operator(img, sigma=2.3, low_threshold=0.45, high_threshold=0.75):
    print("applied Canny")
    if len(img.shape) == 3:
        img = skimage.color.rgb2gray(img)
    #does canny operator, method is Gaussian filtering -> Sobel operator -> Non-maxima supression -> Hysteresis thresholding
    edges = feature.canny(img, sigma=sigma, low_threshold=low_threshold , high_threshold=high_threshold, use_quantiles=True)
    #invert images so identify edges as black
    inverted_edges = np.where(edges.astype(np.uint8) , 0, 255).astype(np.uint8)
    return inverted_edges

def noise_mean_filter(img,mean):
    #Smooths img by convolving a 3x3 kernel with same value to img
    if len(img.shape) == 3:
        img = skimage.color.rgb2gray(img)
    kernel = np.zeros([3,3],dtype=float)
    kernel = np.full_like(kernel,mean)
    smoothed = scipy.ndimage.convolve(img,kernel)
    return smoothed


def noise_gaussian_filter(img,size=3,sigma=1):
    #Smooths image by convolving gaussian kernel with img
    if len(img.shape) == 3:
        img = skimage.color.rgb2gray(img)
    g = get_gaussian_kernel(size,sigma)
    smoothed = scipy.ndimage.convolve(img,g)
    return smoothed

def get_gaussian_kernel(size, sigma=1):
    #returns a gaussian kernel based on argument size and sigma
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def otsu_threshold(img,flip=0):
    #applies otsu thresholding to img, flip is controlled for if img > threshold value
    #it would return 255 or 0
    threshold_value = skimage.filters.threshold_otsu(img)
    mask = img > threshold_value

    if flip:
        final = np.where(mask, 255, 0).astype(np.uint8)
    else:
        final = np.where(mask, 0, 255).astype(np.uint8)
    return final

def roberts_operator(img):
    print("applied Roberts")
    if len(img.shape) == 3:
        img = skimage.color.rgb2gray(img)
    # apply roberts operator on both x and y axis, then calculate magnitude
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])
    roberts_x = scipy.ndimage.convolve(img, kernel_x)
    roberts_y = scipy.ndimage.convolve(img, kernel_y)
    magnitude = np.sqrt(roberts_x ** 2 + roberts_y ** 2)
    magnitude *= 255.0 / np.max(magnitude)

    #apply otsu thresholding
    return otsu_threshold(magnitude)

def sobel_operator(img):
    print("applied Sobel")
    if len(img.shape) == 3:
        img = skimage.color.rgb2gray(img)

    #apply sobel operator on both x and y axis, then calculate magnitude
    sobel_h = scipy.ndimage.sobel(img, axis=0)
    sobel_v = scipy.ndimage.sobel(img, axis=1)
    magnitude = np.sqrt(sobel_h ** 2 + sobel_v ** 2)
    magnitude *= 255.0 / np.max(magnitude)

    # apply otsu thresholding
    return otsu_threshold(magnitude)


def improvised_sobel_operator(img):
    """Special operator
        1)apply Otsu thresholding
        2)Sobel operator
    """
    print("applied Improv")

    if len(img.shape) == 3:
        img = skimage.color.rgb2gray(img)

    threshold_value = skimage.filters.threshold_otsu(img)
    final_img = img > threshold_value

    sobel_h = scipy.ndimage.sobel(final_img, axis=0)
    sobel_v = scipy.ndimage.sobel(final_img, axis=1)
    magnitude = np.sqrt(sobel_h ** 2 + sobel_v ** 2)
    magnitude *= 255.0 / np.max(magnitude)

    binary_img1 = (magnitude == 0).astype(np.uint8)
    return binary_img1


def prewitt_operator(img):
    #test func, didnt use
    if len(img.shape) == 3:
        img = skimage.color.rgb2gray(img)

    prewitt_h = scipy.ndimage.prewitt(img, axis=0)
    prewitt_v = scipy.ndimage.prewitt(img, axis=1)
    magnitude = np.sqrt(prewitt_h ** 2 + prewitt_v ** 2)
    magnitude *= 255.0 / np.max(magnitude)

    return otsu_threshold(magnitude)


def first_order_gaussian_operator(img, sigma=1):
    print("applied First order Gaussian")
    if len(img.shape) == 3:
        img = skimage.color.rgb2gray(img)

    #convolve img with gaussian kernel
    img = scipy.ndimage.gaussian_filter(img, sigma)

    return otsu_threshold(img)



def laplacian_operator(img):
    print("applied Laplacian")
    if len(img.shape) == 3:
        img = skimage.color.rgb2gray(img)

    #convolve img with laplacian kernel
    img = scipy.ndimage.laplace(img)

    return otsu_threshold(img,1)

def laplacian_of_Gaussian_operator(img, sigma=1):
    """Process
        1) produce a gaussian kernel
        2) convolve laplacian kernel with gaussian
        3) convolve image with kernel produced in step 2
    """
    print("applied Laplacian of Gaussian")
    if len(img.shape) == 3:
        img = skimage.color.rgb2gray(img)
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    # get a gaussian kernel 3x3
    gaussian_kernel = get_gaussian_kernel(3, sigma)
    # Laplacian kernel convolve with gaussian kernel, zero padding
    first_convolve = scipy.signal.convolve2d(laplacian_kernel, gaussian_kernel)
    #img convolve with result from above, size will be same as img
    img = scipy.signal.convolve2d(img, first_convolve,mode='same')

    return otsu_threshold(img,1)


def apply_threshold(img,val=127):
    #returns binary image where intensity > 127 is class as 1
    return np.where(img > val, 0, 255).astype(np.uint8)

def zero_cross(image, thresh = 0.75 ):
    #determines zero cross by finding max and min in the 3x3 patch
    #and finding the absolute difference between them, if greater than threshold than it is
    #setting the pixel as 1
    z_c_image = np.zeros(image.shape)
    thresh = np.absolute(image).mean() * thresh
    h,w = image.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            patch = image[y-1:y+2, x-1:x+2]
            p = image[y, x]
            maxP = patch.max()
            minP = patch.min()
            if (p > 0):
                zeroCross = True if minP < 0 else False
            else:
                zeroCross = True if maxP > 0 else False
            if ((maxP - minP) > thresh) and zeroCross:
                z_c_image[y, x] = 1
    return z_c_image


def zero_cross_with_smoothing(image, tune_sigma=3, thresh_factor=0.8):
    # Apply Gaussian smoothing
    smooth_image = scipy.ndimage.gaussian_filter(image, tune_sigma)
    # Perform zero-crossing
    z_c_image = zero_cross(smooth_image, thresh_factor)
    return z_c_image