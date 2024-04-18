import skimage
import scipy
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

from util import show_rgb_image, show_binary_image

def noise_mean_filter(img,mean):
    if len(img.shape) == 3:
        img = skimage.color.rgb2gray(img)
    kernel = np.zeros([3,3],dtype=float)
    kernel = np.full_like(kernel,mean)
    smoothed = scipy.ndimage.convolve(img,kernel)
    return smoothed

def laplacian_of_gaussian(img, sigma=1.0):
    smooth_img = scipy.ndimage.gaussian_filter(img, sigma)
    edge_detection_img = scipy.ndimage.gaussian_laplace(smooth_img, sigma)
    return edge_detection_img


def get_gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def laplacian_of_gaussian_own(img, sigma):
    """Process
    1) produce a gaussian kernel
    2) convolve laplacian kernel with gaussian
    3) convolve image with kernel produced in step 2
    """
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    gaussian_kernel = get_gaussian_kernel(3,sigma)

    # Laplacian kernel convolve with gaussian kernel
    first_convolve = scipy.signal.convolve2d(laplacian_kernel, gaussian_kernel) #same size so dont need fill padding

    print("first", first_convolve)

    # img convolve with first result
    result = scipy.signal.convolve2d(img, first_convolve)

    print("maxmin",np.max(result),np.min(result))

    return result

def noise_gaussian_filter(img,sigma=1,size=3):
    if len(img.shape) == 3:
        img = skimage.color.rgb2gray(img)
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal

    smoothed = scipy.ndimage.convolve(img,g)
    return smoothed


def binarize_image(image, threshold=128):
    return np.where(image > threshold, 255, 0).astype(np.uint8)


def zero_cross(image, thresh = 0.75 ):
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

    # Perform zero-crossing detection on the smoothed image
    z_c_image = zero_cross(smooth_image, thresh_factor)

    return z_c_image

if "__main__":
    shakey = np.squeeze(skimage.io.imread('./Data/shakey.150.gif'))
    tune_sigma = 2.7

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    axes = axes.ravel()

    axes[0].imshow(shakey, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].set_axis_off()

    # Apply the Laplacian of Gaussian filter
    smooth = noise_gaussian_filter(shakey,1.5)
    smooth2 = noise_gaussian_filter(shakey, 1.5)

    log_mask = laplacian_of_gaussian_own(smooth, tune_sigma)

    log_mask2 = laplacian_of_gaussian_own(smooth2, 1)


    log_mask_zeroC = zero_cross_with_smoothing(log_mask)
    binary_log_mask = binarize_image(log_mask2,4)
    print(log_mask)


    axes[1].imshow(laplacian_of_gaussian(shakey, 1.43), cmap='gray')
    axes[1].set_title("LoG (scipy library)")
    axes[1].set_axis_off()

    axes[2].imshow(log_mask_zeroC, cmap='gray') #laplacian_of_gaussian(shakey,tune_sigma)
    axes[2].set_title("own LoG (zero crossing)")
    axes[2].set_axis_off()

    axes[3].imshow(binary_log_mask, cmap='gray')#scipy.ndimage.gaussian_filter(shakey, tune_sigma)
    axes[3].set_title("own LoG (threshold)")
    axes[3].set_axis_off()

    plt.show()


input_array = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])

# Laplacian kernel (3x3)
laplacian_kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])

result = scipy.signal.convolve2d(input_array, laplacian_kernel, mode='same')

print("Input Array:")
print(input_array)
print("\nLaplacian Kernel:")
print(laplacian_kernel)
print("\nResult (Convolution with Zero Padding):")
print(result)