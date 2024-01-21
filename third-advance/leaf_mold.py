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
    img_path = path.join(parent_path, "images", "leaf_mold")

    # Leemos las imágenes
    images = read_images(img_path)

    # Mostramos la ecualización de histograma de las imágenes
    for image, file in images:
        # Convertimos la imagen a escala de grises
        image_gray = ski.color.rgb2gray(image)

        image_gray_byte = ski.util.img_as_ubyte(image_gray)

        grad = ski.filters.rank.enhance_contrast_percentile(
            image_gray_byte, ski.morphology.disk(8), p0=0.15, p1=0.85
        )

        image2 = ski.util.img_as_float64(grad)

        smooth = ski.filters.gaussian(
            image2, sigma=5.5, mode="mirror", preserve_range=True
        )

        thresh_value = ski.filters.threshold_sauvola(smooth, window_size=899, k=0.099)
        thresh = smooth <= thresh_value

        fill = ndi.binary_fill_holes(thresh)
        clear = ski.segmentation.clear_border(fill)

        dilate = ski.morphology.binary_dilation(clear)
        erode = ski.morphology.binary_erosion(clear)

        mask = np.logical_and(dilate, ~erode)

        canny = ski.feature.canny(image_gray, sigma=2.5)

        # Umbralización global, isodata
        isodata_threshold = ski.filters.threshold_isodata(image_gray)
        isodata_filtered_image = image_gray <= isodata_threshold

        # Otsu
        median = ski.filters.median(image_gray)
        threshold_value = ski.filters.threshold_otsu(median)
        otsu_filtered_image = median <= threshold_value

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
        gauss = ski.filters.gaussian(image2,sigma=4, mode = 'reflect')
        thres_otsu_multi = ski.filters.threshold_multiotsu(gauss, classes=3)
        regions_bn = np.digitize(gauss, bins=thres_otsu_multi)

        # Mostrar resultados
        plt.figure()

        plt.suptitle(
            f"{path.basename(file)}",
            y=0.96,
            fontsize=14,
        )

        plt.subplot(2, 4, 1)
        plt.imshow(image, cmap=plt.cm.gray)
        plt.title("Original")

        plt.subplot(2, 4, 2)
        plt.imshow(image2, cmap=plt.cm.gray)
        plt.title("Enhance contrast percentile")

        # Segmentación
        plt.subplot(2, 4, 3)
        plt.imshow(canny, cmap=plt.cm.gray)
        plt.title("Canny")

        plt.subplot(2, 4, 4)
        plt.imshow(mask, cmap=plt.cm.gray)
        plt.title("Umbral de Sauvola")

        plt.subplot(2, 4, 5)
        plt.imshow(isodata_filtered_image, cmap=plt.cm.gray)
        plt.title("Umbralización Global")

        plt.subplot(2, 4, 6)
        plt.imshow(otsu_filtered_image, cmap=plt.cm.gray)
        plt.title("Umbral Otsu")

        plt.subplot(2, 4, 7)
        plt.imshow(regions, cmap="jet")
        plt.title("Multi-Otsu color")

        plt.subplot(2, 4, 8)
        plt.imshow(regions_bn, cmap=plt.cm.binary)
        plt.title("Multi-Otsu b&n")

    plt.show()


if __name__ == "__main__":
    main()
