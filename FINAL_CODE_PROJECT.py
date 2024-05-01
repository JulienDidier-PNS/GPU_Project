import sys
import os
from numba import cuda

#PARTIE BW
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
    from PIL import Image
    import numpy as np
    import math
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