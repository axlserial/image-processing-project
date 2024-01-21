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
    img_path = path.join(parent_path, "images", "septoria")

    # Leemos las imágenes
    images = read_images(img_path)

    # Mostramos la ecualización de histograma de las imágenes
    for image, file in images:
        # Convertimos la imagen a escala de grises
        image_gray = ski.color.rgb2gray(image)
        image_gray = ski.util.img_as_ubyte(image_gray)

        # equalized = ski.exposure.equalize_hist(image_gray)

        # Filtro para reducir sal y pimienta (mediana)
        median_filtered = ski.filters.median(image_gray)

        # Filtro para reducir pimienta (maximo)
        max_filtered = ski.filters.rank.maximum(image_gray, ski.morphology.disk(5))

        # Filtro para reducir sal (minimo)
        min_filtered = ski.filters.rank.minimum(image_gray, ski.morphology.disk(3))

        # Filtro
        sobel_filtered = ski.filters.sobel(median_filtered, mode="constant")

        threshold_value = ski.filters.threshold_otsu(median_filtered)
        otsu_filtered_image = median_filtered <= threshold_value

        canny = ski.feature.canny(image_gray, sigma=1.5, mode="mirror")

        # Umbralización global, isodata
        isodata_threshold = ski.filters.threshold_isodata(image_gray)
        isodata_filtered_image = image_gray <= isodata_threshold

        # Multiple de otsu
        # Obtener los umbrales óptimos
        thresholds = ski.filters.threshold_multiotsu(image_gray)

        # Aplicar los umbrales a la imagen
        regions = np.zeros_like(image_gray)
        for i, threshold in enumerate(thresholds):
            if i == 0:
                regions[image_gray < threshold] = i
            else:
                regions[
                    (image_gray >= thresholds[i - 1]) & (image_gray < threshold)
                ] = i
            regions[image_gray >= thresholds[-1]] = i + 1

        # Prueba con otsu Multiple, 2 clases
        thres_otsu_multi = ski.filters.threshold_multiotsu(median_filtered, classes=3)
        regions_bn = np.digitize(median_filtered, bins=thres_otsu_multi)

        # Mostrar resultados
        plt.figure()

        plt.suptitle(
            f"{path.basename(file)}",
            y=0.96,
            fontsize=14,
        )

        plt.subplot(3, 3, 1)
        plt.imshow(image, cmap=plt.cm.gray)
        plt.title("Original")

        plt.subplot(3, 3, 2)
        plt.imshow(median_filtered, cmap=plt.cm.gray)
        plt.title("Mediana")

        # Segmentación
        plt.subplot(3, 3, 4)
        plt.imshow(sobel_filtered, cmap=plt.cm.gray)
        plt.title("Sobel")

        plt.subplot(3, 3, 5)
        plt.imshow(canny, cmap=plt.cm.gray)
        plt.title("Canny sigma 1.5")

        plt.subplot(3, 3, 6)
        plt.imshow(isodata_filtered_image, cmap=plt.cm.gray)
        plt.title("Umbralización Global")

        plt.subplot(3, 3, 7)
        plt.imshow(otsu_filtered_image, cmap=plt.cm.gray)
        plt.title("Umbral Otsu")

        plt.subplot(3, 3, 8)
        plt.imshow(regions, cmap="jet")
        plt.title("Multi-Otsu color")

        plt.subplot(3, 3, 9)
        plt.imshow(regions_bn, cmap="binary")
        plt.title("Multi-Otsu b&n")

    plt.show()


if __name__ == "__main__":
    main()
