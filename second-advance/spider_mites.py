from os import listdir, path
import skimage as ski
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def read_images(path: str):
    """Función que lee las imágenes de un directorio y las devuelve una a una"""

    for file in listdir(path):
        img = ski.io.imread(f"{path}/{file}")
        yield img, file


def main():
    # Obtenemos la ruta absoluta del archivo actual
    dir_path = path.dirname(path.realpath(__file__))

    # Leemos las imágenes
    images = read_images(path.join(dir_path, "spider_mites"))

    # Mostramos la ecualización de histograma de las imágenes
    for image, file in images:
        # Convertimos la imagen a escala de grises
        image_gray = ski.color.rgb2gray(image)
        image_gray = ski.util.img_as_ubyte(image_gray)

        equalized = ski.exposure.equalize_hist(image_gray)

        median_filtered = ski.exposure.adjust_gamma(equalized, gamma=1.5)

        median = ski.filters.median(image_gray)
        threshold_value = ski.filters.threshold_otsu(median)
        otsu_filtered_image = median <= threshold_value     

       
        # Mostrar resultados
        plt.figure()

        plt.suptitle(
            f"{path.basename(file)}",
            y=0.96,
            fontsize=14,
        )

        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap=plt.cm.gray)
        plt.title("Original")

        plt.subplot(1, 3, 2)
        plt.imshow(median_filtered, cmap=plt.cm.gray)
        plt.title("Gamma 1.5")

        plt.subplot(1, 3, 3)
        plt.imshow(otsu_filtered_image, cmap=plt.cm.gray)
        plt.title("Bordes")



    plt.show()


if __name__ == "__main__":
    main()
