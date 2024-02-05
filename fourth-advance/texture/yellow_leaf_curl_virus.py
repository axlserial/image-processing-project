import skimage as ski
import numpy as np

from tabulate import tabulate

from utils import read_images


def main():
    # Leemos las imágenes
    images = read_images(__file__, "yellow_leaf_curl_virus")

    # Procesamos las imágenes
    data = []
    for image, file in images:
        # Convertimos la imagen a escala de grises
        image_gray = ski.color.rgb2gray(image)

        # Filtros
        equalized = ski.exposure.equalize_hist(image_gray)
        median_filtered = ski.exposure.adjust_gamma(equalized, gamma=2)
        image_filtered = ski.filters.median(median_filtered)
        sobel_filtered = ski.filters.sobel(image_filtered)

        # Aplicamos el método de Otsu multiple
        gauss = ski.filters.gaussian(
            sobel_filtered, sigma=3, mode="mirror", preserve_range=True
        )
        thres_otsu_multi = ski.filters.threshold_multiotsu(gauss, classes=3)
        otsu_multi_res = np.digitize(gauss, bins=thres_otsu_multi)

        # ---------------------------------------------------------------------------- #

        # -- Textura / Estadísticos de primer orden --

        # Obtenemos la media de los niveles de intensidad
        mean = np.mean(otsu_multi_res)

        # Obtenemos la desviación estándar de los niveles de intensidad
        std = 1 - (1 / (1 + np.var(otsu_multi_res, ddof=1)))

        # Obtenemos el valor de uniformidad

        # -- Obtenemos P(i), la probabilidad de ocurrencia de cada nivel de intensidad
        hist, _ = ski.exposure.histogram(otsu_multi_res)
        p_i = hist / np.sum(hist)

        # -- Con el valor de P(i), obtenemos la uniformidad
        uniformity = np.sum(p_i**2)

        # Obtenemos la entropía de la imagen
        entropy = ski.measure.shannon_entropy(otsu_multi_res)

        # Guardamos los datos
        data.append(
            {
                "Archivo": file,
                "Media": f"{mean:.4f}",
                "Desv. Estándar": f"{std:.4f}",
                "Uniformidad": f"{uniformity:.4f}",
                "Entropía": f"{entropy:.4f}",
            }
        )

    # Mostramos los resultados
    print("\nDescriptores de textura de primer orden: Yellow Leaf Curl Virus")
    print(tabulate(data, headers="keys", tablefmt="pretty"))


if __name__ == "__main__":
    main()
