import sys
import os
import time
from numba import cuda
import numpy as np
from PIL import Image
import math

#PARTIE UTILS

def saveImage(output,path):
    final_image = Image.fromarray(output)
    print(path)
    final_image.save(path)

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

def compute_BW(img_src,tb_to_use):
    img = Image.open(img_src)
    src = np.array(img)
    #On range les données en mémoire de manière contigues (sans espaces entre-elles)
    #Conseillé dans le cours
    src_contigous = np.ascontiguousarray(src)
    #on prends les dimensions de l'image
    height, width, _ = src.shape
    #un thread par pixel
    block_size = tb_to_use
    #On calcule la taille de notre grille --> taille de l'image
    grid_size = (math.ceil(height / block_size[0]), math.ceil(width / block_size[1]))

    # on copie le tableau contigue
    input_cuda_rgb_img = cuda.to_device(src_contigous)
    output_cuda_gray_img = cuda.device_array((height, width))

    # On compute l'image
    computeBWcuda[grid_size, block_size](input_cuda_rgb_img, output_cuda_gray_img)

    #on copie le résultat vers le cpu
    grayscale_image = output_cuda_gray_img.copy_to_host()

    #grayscale_pil_image.save(new_filepath)
    return grayscale_image.astype(np.uint8)

################ FIN PARTIE BW ################

################ PARTIE GAUSS ################

@cuda.jit
def gaussian_blur_kernel(input_image, output_image, filter, filter_sum):
    x, y = cuda.grid(2)

    rows, cols = input_image.shape
    if x >= rows or y >= cols:
        return  # Vérifier les limites pour éviter des accès hors des limites de l'image

    filter_size = filter.shape[0]
    radius = filter_size // 2

    temp = 0.0
    for i in range(filter_size):
        for j in range(filter_size):
            offset_x = x + i - radius
            offset_y = y + j - radius
            # Gérer les bords de l'image
            if 0 <= offset_x < rows and 0 <= offset_y < cols:
                temp += input_image[offset_x, offset_y] * filter[i, j]
            else:
                temp += input_image[x, y] * filter[i, j]
        
        # Appliquer la somme du filtre et saturer les valeurs pour qu'elles restent dans [0, 255]
        output_image[x, y] = min(max(int(temp / filter_sum), 0), 255)

def apply_gaussian_blur(image,tb_to_use):
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
    threadsperblock = tb_to_use
    blockspergrid_x = (input_image_np.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (input_image_np.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Lancer le kernel
    gaussian_blur_kernel[blockspergrid, threadsperblock](
        input_image_gpu, output_image_gpu, gaussian_filter_gpu, filter_sum
    )

    # Récupérer les résultats
    output_image_np = output_image_gpu.copy_to_host()
    return output_image_np

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
        #MAXIMUM 175 si la valeur est > 255
        gx = 175 if gx>255 else gx
        gy = 175 if gy>255 else gy
        output_magnitude[y, x] = math.sqrt(gx**2 + gy**2)
        output_direction[y, x] = math.atan2(gy, gx)

def apply_sobel_filter(image,tb_to_use):
    image_np = image

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
    
    threadsperblock = tb_to_use
    blockspergrid_x = int(np.ceil(image_np.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(image_np.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    sobel_kernel[blockspergrid, threadsperblock](image_gpu, magnitude_gpu, direction_gpu, Sx_gpu, Sy_gpu)
    
    magnitude = magnitude_gpu.copy_to_host()
    direction = direction_gpu.copy_to_host()

    return magnitude,direction

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

def apply_threshold(magnitude,tb_to_use):
    output = np.zeros_like(magnitude, dtype=np.uint8)
    magnitude_gpu = cuda.to_device(magnitude)
    output_gpu = cuda.device_array_like(output)

    threadsperblock = tb_to_use
    blockspergrid_x = (magnitude.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (magnitude.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]

    # Seuils pour le thresholding
    low_thresh = 51
    high_thresh = 102

    threshold_kernel[(blockspergrid_x, blockspergrid_y), threadsperblock](
        magnitude_gpu, output_gpu, low_thresh, high_thresh
    )

    output = output_gpu.copy_to_host()
    return output
################ FIN PARTIE THRESHOLD ################

################ PARTIE HYSTERESIS ################

@cuda.jit
def hysteresis_kernel(thresholded_image, output):
    x, y = cuda.grid(2)
    rows, cols = thresholded_image.shape
    if x >= rows or y >= cols:
        return

    # Ne traiter que les pixels faibles
    if thresholded_image[x, y] == 25:
        # Vérifier les 8 voisins pour voir si au moins un est un bord fort
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if thresholded_image[nx, ny] == 255:
                        output[x, y] = 255
                        return
        output[x, y] = 0
    else:
        # Copier la valeur de l'entrée vers la sortie pour les bords forts et non-bords
        output[x, y] = thresholded_image[x, y]

def apply_hysteresis(image,tb_to_use):
    thresholded_image = np.array(image)

    output = np.zeros_like(thresholded_image)
    thresholded_image_gpu = cuda.to_device(thresholded_image)
    output_gpu = cuda.device_array_like(output)

    threadsperblock = tb_to_use
    blockspergrid_x = (thresholded_image.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (thresholded_image.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]

    hysteresis_kernel[(blockspergrid_x, blockspergrid_y), threadsperblock](
        thresholded_image_gpu, output_gpu
    )

    final_output = output_gpu.copy_to_host()
    return final_output
################ FIN PARTIE HYSTERESIS ################

def main():
    print(sys.argv)
    #savoir si la commmande contient 'help'
    if '--help' in sys.argv:
        print("Voici l'utilisation de ce programme:")
    else:
        print("NBArguments : ",len(sys.argv))
        if len(sys.argv) < 3 :
            print("Vous devez spécifier une image source et un chemin de destination !!")
            return -1
        
        if len(sys.argv) > 5 :
            print("Il y a trop d'options 🤯🤯")
            return -1

        img_src = sys.argv[len(sys.argv)-2]
        print("IMG SRC : ",img_src)
        img_dst = sys.argv[len(sys.argv)-1]
        print("IMG DST : ",img_dst)

        if('--tb' in sys.argv):
            tb_to_use = (
                int(sys.argv[sys.argv.index('--tb')+1]),
                int(sys.argv[sys.argv.index('--tb')+1])
                )
        else:
            tb_to_use = (16,16)

        start = time.time()

        if('--threshold' in sys.argv):
            print("perform all kernels up to threshold_kernel")
            output = compute_BW(img_src,tb_to_use)
            output = apply_gaussian_blur(output,tb_to_use)
            magnitude, direction = apply_sobel_filter(output,tb_to_use)
            final_output = apply_threshold(magnitude,tb_to_use)
            saveImage(final_output,img_dst)
        elif('--sobel' in sys.argv):
            print("perform all kernels up to sobel_kernel and write to disk the magnitude of each pixel")
            output = compute_BW(img_src,tb_to_use)
            output = apply_gaussian_blur(output,tb_to_use)
            final_output,direction = apply_sobel_filter(output,tb_to_use)
            saveImage(final_output,img_dst)
        elif('--gauss' in sys.argv):
            print("perform the bw_kernel and the gauss_kernel")
            img_output = compute_BW(img_src,tb_to_use)
            final_output = apply_gaussian_blur(img_output,tb_to_use)
            saveImage(final_output,img_dst)
        elif('--bw' in sys.argv):
            print('Black and White computing..')
            final_output = compute_BW(img_src,tb_to_use)
            saveImage(final_output,img_dst)
        else:
            print("performing all kernels")
            output = compute_BW(img_src,tb_to_use)
            output = apply_gaussian_blur(output,tb_to_use)
            magnitude, direction = apply_sobel_filter(output,tb_to_use)
            output = apply_threshold(magnitude,tb_to_use)
            final_output = apply_hysteresis(output,tb_to_use)
            saveImage(final_output,img_dst)
        stop = time.time()
        elapsed = stop-start
        print(f'execution time : {elapsed:.3} ms')


main()