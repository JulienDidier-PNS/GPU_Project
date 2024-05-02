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

################ FIN PARTIE GAUSS ################

################ PARTIE SOBEL ################

@cuda.jit
def sobel_kernel(image, output, Sx, Sy):
    i, j = cuda.grid(2)
    if i > 0 and i < image.shape[0] - 1 and j > 0 and j < image.shape[1] - 1:
        gx = 0
        gy = 0
        for k in range(-1, 2):
            for l in range(-1, 2):
                gx += image[i + k, j + l] * Sx[k + 1, l + 1]
                gy += image[i + k, j + l] * Sy[k + 1, l + 1]
        output[i, j] = math.sqrt(gx**2 + gy**2)

def apply_sobel_filter(image_src):
    # On charge l'image et on la convertit en tableau NumPy
    image = Image.open(image_src).convert('L') # CE CONVERT DEVRA POTENTIELLEMENT SAUTER
    image_np = np.array(image)

    # On créé nos tableaux d'entrée et de sortie sur le GPU
    image_gpu = cuda.to_device(image_np)
    output_gpu = cuda.device_array_like(image_np)

    # On définit nos matrices de convolution comme des constantes
    Sx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    Sy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
    
    # On transfère Sx et Sy sur le GPU (sans ces deux lignes, on se prend un warning 'NumbaPerformanceWarning')
    Sx_gpu = cuda.to_device(Sx)  
    Sy_gpu = cuda.to_device(Sy)
    
    # On définit les dimensions du bloc et de la grille
    threadsperblock = (16, 16) # D'après mes recherches, un block de 256 threads semble optimisé, permettant de maintenir une haute occupation des ressources du GPU sans les surcharger
    blockspergrid_x = int(np.ceil(image_np.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(image_np.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    # On lance le kernel avec les matrices sur le GPU
    sobel_kernel[blockspergrid, threadsperblock](image_gpu, output_gpu, Sx_gpu, Sy_gpu)
    
    # On récupère le résultat du GPU pour le copier sur le CPU, et on enregistre l'image
    sobel_image_np = output_gpu.copy_to_host()
    sobel_image = Image.fromarray(sobel_image_np.astype(np.uint8))
    sobel_image.save('sobel_image_numba.jpg')

    print("Fin du filtre de Sobel avec GPU utilisant Numba")

################ FIN PARTIE SOBEL ################

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