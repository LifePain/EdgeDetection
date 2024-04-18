import skimage
import scipy
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.metrics import confusion_matrix,classification_report,f1_score , accuracy_score


import myownlib

images = {
    "set1_img1": skimage.color.rgb2gray(skimage.io.imread('./Data/cells/9343 AM.bmp')),
    "set1_img2": skimage.io.imread('./Data/cells/9343 AM Edges.bmp'),
    "set2_img1": skimage.color.rgb2gray(skimage.io.imread('./Data/cells/10905 JL.bmp')),
    "set2_img2": skimage.io.imread('./Data/cells/10905 JL Edges.bmp'),
    "set3_img1": skimage.color.rgb2gray(skimage.io.imread('./Data/cells/43590 AM.bmp')),
    "set3_img2": skimage.io.imread('./Data/cells/43590 AM Edges.bmp')
}

def show_binary_image(image, title=None):
    plt.imshow(image,cmap='gray')

    # remove the axis / ticks for a clean looking image
    plt.xticks([])
    plt.yticks([])

    # if a title is provided, show it
    if title is not None:
        plt.title(title)

    plt.show()

def plot_operators(axis,row,text,  operator_func):
    for i in range(3):
        key = f"set{i + 1}_img1"
        img = images[key]

        processed_img = operator_func(myownlib.noise_gaussian_filter(img, 3, 1.5))
        #processed_img = myownlib.apply_threshold(operator_func(myownlib.noise_gaussian_filter(img, 3, 1.5)),threshold)
        if i == 1:
            axs[row][i].set_title(text)

        axis[row][i].imshow(processed_img, cmap='gray', aspect='equal')
        axis[row][i].set_axis_off()

ROC_arr = []


def plot_ROC(ROC_arr):
    # Define colors for each operator
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    edgeDetectors = ['Roberts','Sobel','First Order Gaussian','Laplacian','Laplacian of Gaussian','Canny','*Special']

    # Plot ROC curve for each edge detector
    for i, edge_detector_data in enumerate(ROC_arr, start=1):
        operator_color = colors[i-1]
        for j, (specificity, sensitivity) in enumerate(edge_detector_data):
            plt.plot(1 - specificity, sensitivity, 'o', color=operator_color,
            label=f'{edgeDetectors[i-1]}' if j == 0 else None)

    # Plot random classifier line (diagonal)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')

    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) space')
    plt.legend()
    plt.grid(True)
    plt.show()


def get_set_conf_matrix(operator_func):
    arr = []
    avg_f1,avg_acc = 0,0
    for i in range(3):
        key_ori = f"set{i + 1}_img1"
        key_GT = f"set{i + 1}_img2"
        img_ori = images[key_ori]
        GT_img = skimage.color.rgb2gray(images[key_GT])

        img_ori = operator_func(myownlib.noise_gaussian_filter(img_ori, 3, 1.5))
        arr.append(get_confucion_matrix(img_ori,GT_img))
        f1,acc = get_acc(img_ori,GT_img)
        avg_f1 += f1
        avg_acc += acc
    avg_f1 /= 3
    avg_acc /= 3

    print(f"AVG f1 score : {avg_f1}")
    print(f"AVG accuracy : {avg_acc}")
    return arr

def get_acc(img,GT_img):
    binary_img1 = img
    if(np.max(img) == 255):
        binary_img1 = (img > 127).astype(np.uint8)
    binary_img2 = (GT_img > 0.5).astype(np.uint8)
    binary_img1 = binary_img1.flatten()
    binary_img2 = binary_img2.flatten()

    f1 = f1_score(binary_img1,binary_img2)
    acc = accuracy_score(binary_img1, binary_img2)



    return (f1,acc)


def get_confucion_matrix(img,GT_img):
    #binary_img1 = (img == 0).astype(np.uint8)
    binary_img1 = img
    if(np.max(img) == 255):
        binary_img1 = (img > 127).astype(np.uint8)
    binary_img2 = (GT_img > 0.5).astype(np.uint8)

    # print("[DEBUG] binary image 1")
    # print(f"-shape : {np.shape(binary_img1)} \n -max/min {np.max(binary_img1)},{np.min(binary_img1)}")
    # print("[DEBUG] binary image 2")
    # print(f"-shape : {np.shape(binary_img2)} \n -max/min {np.max(binary_img2)},{np.min(binary_img2)}")

    #to print
    #---------
    # fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    # axs = axs.ravel()
    # axs[0].imshow(binary_img2, cmap='gray', aspect='equal')
    # axs[1].imshow(binary_img1, cmap='gray', aspect='equal')
    # axs[0].set_axis_off()
    # axs[1].set_axis_off()
    # plt.tight_layout()
    # plt.show()
    #-----

    binary_img1 = binary_img1.flatten()
    binary_img2 = binary_img2.flatten()


    conf_matrix = confusion_matrix(binary_img1, binary_img2)
    tn, fp, fn, tp = conf_matrix.ravel()
    # print("conf_matrix",conf_matrix)
    #
    # print("f1 score : ",f1_score(binary_img1,binary_img2))
    # print("accuraccy : ", accuracy_score(binary_img1, binary_img2))

    sensitivity = tp / (tp + fn)  # recall
    specificity = tn / (tn + fp)
    return (specificity,sensitivity)



fig, axs = plt.subplots(4, 3, figsize=(12, 12))  # 7,3

#display original and ground truth (row 0,1)
for i in range(6):
    row = i % 2
    col = i // 2
    key = f"set{col + 1}_img{row + 1}"
    img = images[key]
    axs[row, col].imshow(img, cmap='gray', aspect='equal')
    axs[row, col].set_axis_off()
    if col == 1:
        if row == 0:
            axs[row, col].set_title("Original Image")
        else:
            axs[row, col].set_title("Ground Truth")


##for i in range(3):
##    col = i
##    key = f"set{col+ 1}_img2"
##    img = images[key]
##    axs[0, col].imshow(img, cmap='gray', aspect='equal')
##    axs[0, col].set_axis_off()
##    if col == 1:
##        axs[0, col].set_title("Ground Truth")


## display roberts and sobel
#plot_operators(axs,1,"Roberts",myownlib.roberts_operator)
plot_operators(axs,2,"Special",myownlib.improvised_sobel_operator)

#plot_operators(axs,1,"Prewitt",myownlib.prewitt_operator)
#plot_operators(axs,2,"Sobel",myownlib.sobel_operator)
#plot_operators(axs,3,"first Order Gaussian",myownlib.first_order_gaussian_operator)
#plot_operators(axs,3,"Laplacian",myownlib.laplacian_operator)
#plot_operators(axs,3,"Laplacian of Gaussian",myownlib.laplacian_of_Gaussian_operator)
plot_operators(axs,3,"Canny",myownlib.canny_edge_operator)
plt.tight_layout()
plt.show()

print("-"*25)
ROC_arr.append(get_set_conf_matrix(myownlib.roberts_operator))
ROC_arr.append(get_set_conf_matrix(myownlib.sobel_operator))
ROC_arr.append(get_set_conf_matrix(myownlib.first_order_gaussian_operator))
ROC_arr.append(get_set_conf_matrix(myownlib.laplacian_operator))
ROC_arr.append(get_set_conf_matrix(myownlib.laplacian_of_Gaussian_operator))
ROC_arr.append(get_set_conf_matrix(myownlib.canny_edge_operator))
ROC_arr.append(get_set_conf_matrix(myownlib.improvised_sobel_operator))
print(ROC_arr)
plot_ROC(ROC_arr)