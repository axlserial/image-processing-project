from os import listdir, path
import skimage as ski
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.ndimage as ndi


def read_images(path: str):
    """Función que lee las imágenes de un directorio y las devuelve una a una"""

    for file in listdir(path):
        img = ski.io.imread(f"{path}/{file}")
        yield img, file


def main():
     # Obtenemos la ruta absoluta del archivo actual
    dir_path = path.dirname(path.realpath(__file__))


    # Leemos las imágenes
    images = read_images(path.join(dir_path, "leaf_mold"))


    # Mostramos la ecualización de histograma de las imágenes
    for image, file in images:
        # Convertimos la imagen a escala de grises
        image_gray = ski.color.rgb2gray(image)
        
        image_gray_byte = ski.util.img_as_ubyte(image_gray)
        
        grad = ski.filters.rank.enhance_contrast_percentile(image_gray_byte, ski.morphology.disk(8),p0=.15, p1=.85)

        image2 = ski.util.img_as_float64(grad)
       
        smooth = ski.filters.gaussian(image2,sigma=5.5, mode = 'mirror', preserve_range = True)       


        thresh_value = ski.filters.threshold_sauvola(smooth, window_size=899, k=0.099)
        thresh = smooth <= thresh_value

        fill = ndi.binary_fill_holes(thresh)
        clear = ski.segmentation.clear_border(fill)

        dilate = ski.morphology.binary_dilation(clear)
        erode = ski.morphology.binary_erosion(clear)
        
        mask = np.logical_and(dilate, ~erode)

        canny = ski.feature.canny(image_gray, sigma=2.5)
        
        #Multiple de otsu
            #Obtener los umbrales óptimos
        thresholds = ski.filters.threshold_multiotsu(image_gray)

        # Aplicar los umbrales a la imagen
        regions = np.zeros_like(image_gray)
        for i, threshold in enumerate(thresholds):
            if i == 0:
                regions[image_gray < threshold] = i
            else:
                regions[(image_gray >= thresholds[i-1]) & (image_gray < threshold)] = i
            regions[image_gray >= thresholds[-1]] = i+1

        

        # Mostrar resultados
        plt.figure()

        plt.suptitle(
            f"{path.basename(file)}",
            y=0.96,
            fontsize=14,
        )

        plt.subplot(2, 3, 1)
        plt.imshow(image, cmap=plt.cm.gray)
        plt.title("Original")

        plt.subplot(2, 3, 2)
        plt.imshow(image2, cmap=plt.cm.gray)
        plt.title("Enhance contrast percentile")

        # Segmentación
        plt.subplot(2, 3, 3)
        plt.imshow(mask, cmap=plt.cm.gray)
        plt.title("Umbral de Sauvola")

        plt.subplot(2, 3, 4)
        plt.imshow(canny, cmap=plt.cm.gray)
        plt.title("Canny")

        plt.subplot(2, 3, 5)
        plt.imshow(regions, cmap='jet')
        plt.title("Multi-Otsu")

        # plt.subplot(1, 3, 3)
        # plt.imshow(otsu_filtered_image, cmap=plt.cm.gray)
        # plt.title("Bordes")


    plt.show()


if __name__ == "__main__":
    main()
