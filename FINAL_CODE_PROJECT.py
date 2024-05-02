import sys
import os
from numba import cuda
import numpy as np
from PIL import Image
import math

################ PARTIE BW ################

@cuda.jit
def computeBWcuda(img,dst):
    #on prend l'indice global
    x,y = cuda.grid(2)
    #On verifie si on est pas hors limite
    if x < dst.shape[0] and y < dst.shape[1]:
        #on compute
        red = img[x, y, 0]
        green = img[x, y, 1]
        blue = img[x, y, 2]
        gray = 0.3 * red + 0.59 * green + 0.11 * blue
        dst[x, y] = gray

def compute_BW(img_src,img_dst):
    img = Image.open(img_src)
    src = np.array(img)
    #On range les données en mémoire de manière contigues (sans espaces entre-elles)
    #Conseillé dans le cours
    src_contigous = np.ascontiguousarray(src)
    #on prends les dimensions de l'image
    height, width, _ = src.shape
    #un thread par pixel
    block_size = (1,1)
    #On calcule la taille de notre grille --> taille de l'image
    grid_size = (math.ceil(height / block_size[0]), math.ceil(width / block_size[1]))

    # on copie le tableau contigue
    input_cuda_rgb_img = cuda.to_device(src_contigous)
    output_cuda_gray_img = cuda.device_array((height, width))

    # On compute l'image
    computeBWcuda[grid_size, block_size](input_cuda_rgb_img, output_cuda_gray_img)

    #on copie le résultat vers le cpu
    grayscale_image = output_cuda_gray_img.copy_to_host()

    # on enregistre l'image
    grayscale_pil_image = Image.fromarray(grayscale_image.astype(np.uint8))
    grayscale_pil_image.save(img_dst)

################ FIN PARTIE BW ################

################ PARTIE GAUSS ################

@cuda.jit
def gaussian_blur_kernel(input_image, output_image, filter, filter_sum):
    x, y = cuda.grid(2)
    rows, cols, channels = input_image.shape
    if x >= rows or y >= cols:
        return  # Vérifier les limites pour éviter des accès hors des limites de l'image

    filter_size = filter.shape[0]
    radius = filter_size // 2

    for c in range(channels):
        temp = 0.0
        for i in range(filter_size):
            for j in range(filter_size):
                offset_x = x + i - radius
                offset_y = y + j - radius
                # Gérer les bords de l'image
                if 0 <= offset_x < rows and 0 <= offset_y < cols:
                    temp += input_image[offset_x, offset_y, c] * filter[i, j]
                else:
                    temp += input_image[x, y, c] * filter[i, j]
        
        # Appliquer la somme du filtre et saturer les valeurs pour qu'elles restent dans [0, 255]
        output_image[x, y, c] = min(max(int(temp / filter_sum), 0), 255)

def apply_gaussian_blur(image_path):
    image = Image.open(image_path)
    input_image_np = np.array(image)

    # Définir le filtre gaussien et sa somme
    gaussian_filter = np.array([[1, 4, 6, 4, 1],
                                [4, 16, 24, 16, 4],
                                [6, 24, 36, 24, 6],
                                [4, 16, 24, 16, 4],
                                [1, 4, 6, 4, 1]], dtype=np.float32)
    filter_sum = gaussian_filter.sum()

    gaussian_filter_gpu = cuda.to_device(gaussian_filter)

    output_image_np = np.zeros_like(input_image_np)

    # Convertir les données en GPU
    input_image_gpu = cuda.to_device(input_image_np)
    output_image_gpu = cuda.device_array_like(input_image_np)

    # Configuration des blocs et des grilles
    threadsperblock = (16, 16)
    blockspergrid_x = (input_image_np.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (input_image_np.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Lancer le kernel
    gaussian_blur_kernel[blockspergrid, threadsperblock](
        input_image_gpu, output_image_gpu, gaussian_filter_gpu, filter_sum
    )

    # Récupérer les résultats
    output_image_np = output_image_gpu.copy_to_host()
    filtered_image = Image.fromarray(output_image_np)
    filtered_image.save('gpu_filtered_image.jpg')

################ FIN PARTIE GAUSS ################

################ PARTIE SOBEL ################

@cuda.jit
def sobel_kernel(image, output_magnitude, output_direction, Sx, Sy):
    y, x = cuda.grid(2)
    rows, cols = image.shape
    if y > 0 and y < rows - 1 and x > 0 and x < cols - 1:
        gx = 0.0
        gy = 0.0
        for i in range(-1, 2):
            for j in range(-1, 2):
                gx += image[y + i, x + j] * Sx[i + 1, j + 1]
                gy += image[y + i, x + j] * Sy[i + 1, j + 1]
        output_magnitude[y, x] = math.sqrt(gx**2 + gy**2)
        output_direction[y, x] = math.atan2(gy, gx)

def apply_sobel_filter(image_src):
    image = Image.open(image_src).convert('L')
    image_np = np.array(image)

    image_gpu = cuda.to_device(image_np)
    magnitude_gpu = cuda.device_array_like(image_np)
    direction_gpu = cuda.device_array_like(image_np)

    Sx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    Sy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
    
    Sx_gpu = cuda.to_device(Sx)
    Sy_gpu = cuda.to_device(Sy)
    
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(image_np.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(image_np.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    sobel_kernel[blockspergrid, threadsperblock](image_gpu, magnitude_gpu, direction_gpu, Sx_gpu, Sy_gpu)
    
    magnitude = magnitude_gpu.copy_to_host()
    direction = direction_gpu.copy_to_host()
    
    return magnitude, direction

################ FIN PARTIE SOBEL ################

################ PARTIE THRESHOLD ################

@cuda.jit
def threshold_kernel(magnitude, output, low_thresh, high_thresh):
    x, y = cuda.grid(2)
    if x < magnitude.shape[0] and y < magnitude.shape[1]:
        mag = magnitude[x, y]
        if mag > high_thresh:
            output[x, y] = 255  # Bord fort
        elif mag > low_thresh:
            output[x, y] = 25   # Bord faible
        else:
            output[x, y] = 0    # Non-bord

def apply_threshold(magnitude):
    output = np.zeros_like(magnitude, dtype=np.uint8)
    magnitude_gpu = cuda.to_device(magnitude)
    output_gpu = cuda.device_array_like(output)

    threadsperblock = (16, 16)
    blockspergrid_x = (magnitude.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (magnitude.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]

    # Seuils pour le thresholding
    low_thresh = 60
    high_thresh = 100

    threshold_kernel[(blockspergrid_x, blockspergrid_y), threadsperblock](
        magnitude_gpu, output_gpu, low_thresh, high_thresh
    )

    output = output_gpu.copy_to_host()
    return output

################ FIN PARTIE THRESHOLD ################

def main():
    print(sys.argv)
    #savoir si la commmande contient 'help'
    if '--help' in sys.argv:
        print("Voici l'utilisation de ce programme:")
    else:
        #savoir si la commande contient 'inputImage' et 'outputImage'
        if '--inputImage' in sys.argv and '--outputImage' in sys.argv:
            print("inputImage and outputImage")
            if('--threshold ' in sys.argv):
                print("perform all kernels up to threshold_kernel")
            elif('--sobel ' in sys.argv):
                print("perform all kernels up to sobel_kernel  and write to disk the magnitude of each pixel")
            elif('--gauss' in sys.argv):
                print("perform the bw_kernel and the gauss_kernel")
                compute_BW(sys.argv[sys.argv.index('--inputImage')+1],sys.argv[sys.argv.index('--outputImage')+1])
            elif('--bw' in sys.argv):
                print('Black and White computing..')
                compute_BW(sys.argv[sys.argv.index('--inputImage')+1],sys.argv[sys.argv.index('--outputImage')+1])
            else:
                print("perform all kernels")
        else:
            #
            print("Les options : inputImage or outputImage sont obligatoires !!")

main()