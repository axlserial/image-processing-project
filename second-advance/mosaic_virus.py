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
    current_path = path.dirname(path.realpath(__file__))
    parent_path = path.dirname(current_path)
    img_path = path.join(parent_path, "images", "mosaic_virus")

    # Leemos las imágenes
    images = read_images(img_path)


    # Mostramos la ecualización de histograma de las imágenes
    for image, file in images:
        # Convertimos la imagen a escala de grises
        image_gray = ski.color.rgb2gray(image)
        image_gray = ski.util.img_as_ubyte(image_gray)

        equalized = ski.exposure.equalize_hist(image_gray)

        # Filtro para reducir sal y pimienta (mediana)
        median_filtered = ski.filters.median(image_gray)

        # Filtro para reducir pimienta (maximo)
        max_filtered = ski.filters.rank.maximum(image_gray, ski.morphology.disk(3))

        # Filtro para reducir sal (minimo)
        min_filtered = ski.filters.rank.minimum(image_gray, ski.morphology.disk(3))

        median = ski.filters.median(image_gray)
        # Filtro
        #sobel_filtered = ski.filters.sobel(ski.filters.median(median))

        canny = ski.feature.canny(median, sigma=3)


        #r = image[:, :, 0]
        #edges = feature.canny(image, sigma=1, low_threshold=10, high_threshold=50)
        #canny = ski.feature.canny(r, sigma=1, low_threshold=10, high_threshold=100)
  

        grad = ski.filters.rank.enhance_contrast_percentile(image_gray, ski.morphology.disk(8),p0=.15, p1=.85)

        image = ski.util.img_as_float64(grad)
        # gauss = ski.filters.gaussian(image_gray,sigma=3, mode = 'reflect', preserve_range = True)
        smooth = ski.filters.gaussian(image,sigma=5.5, mode = 'mirror', preserve_range = True)
      

        thresh_value = ski.filters.threshold_sauvola(smooth, window_size=899, k=0.099)
        thresh = smooth <= thresh_value

        fill = ndi.binary_fill_holes(thresh)
        clear = ski.segmentation.clear_border(fill)

        dilate = ski.morphology.binary_dilation(clear)
        erode = ski.morphology.binary_erosion(clear)
        
        mask = np.logical_and(dilate, ~erode)
       

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
        plt.imshow(median, cmap=plt.cm.gray)
        plt.title("Mediana")

        plt.subplot(2, 3, 3)
        plt.imshow(grad, cmap=plt.cm.gray)
        plt.title("Mejora contraste percentil")

        plt.subplot(2, 3, 4)
        plt.imshow(smooth, cmap=plt.cm.gray)
        plt.title("Gaussiano")
        
        # Segmentación
        plt.subplot(2, 3, 5)
        plt.imshow(canny, cmap=plt.cm.gray)
        plt.title("Canny sigma 3")

        plt.subplot(2, 3, 6)
        plt.imshow(mask, cmap=plt.cm.gray)
        plt.title("Umbral Sauvola")

    plt.show()


if __name__ == "__main__":
    main()
