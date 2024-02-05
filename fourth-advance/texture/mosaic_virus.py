import skimage as ski
import numpy as np

from tabulate import tabulate

from utils import read_images


def main():
    # Leemos las imágenes
    images = read_images(__file__, "mosaic_virus")

    # Procesamos las imágenes
    data = []
    for image, file in images:
        # Convertimos la imagen a escala de grises
        image_gray = ski.util.img_as_ubyte(ski.color.rgb2gray(image))

        # Para aumentar el contraste de la imagen, aplicamos un filtro de realce
        grad = ski.filters.rank.enhance_contrast_percentile(
            image_gray, ski.morphology.disk(8), p0=0.15, p1=0.85
        )
        image2 = ski.util.img_as_float64(grad)

        # Aplicamos el método de Otsu multiple
        gauss = ski.filters.gaussian(
            image2, sigma=4, mode="mirror", preserve_range=True
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
    print("\nDescriptores de textura de primer orden: Mosaic Virus")
    print(tabulate(data, headers="keys", tablefmt="pretty"))


if __name__ == "__main__":
    main()
