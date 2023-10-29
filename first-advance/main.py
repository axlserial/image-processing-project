from os import listdir, path
from skimage import io, exposure, color

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def read_images(path: str):
    """Función que lee las imágenes de un directorio y las devuelve una a una"""

    for file in listdir(path):
        img = io.imread(f"{path}/{file}")
        yield img


def main():
    # Obtenemos la ruta absoluta del archivo actual
    dir_path = path.dirname(path.realpath(__file__))

    # Leemos las imágenes
    images = read_images(path.join(dir_path, "images"))

    # Mostramos la ecualización de histograma de las imágenes
    for i, image in enumerate(images, start=1):
        # Convertimos la imagen a escala de grises
        image_gray = color.rgb2gray(image)

        # Obtiene los histogramas para cada canal de color
        hist_r, bins_r = exposure.histogram(image[:, :, 0])
        hist_g, bins_g = exposure.histogram(image[:, :, 1])
        hist_b, bins_b = exposure.histogram(image[:, :, 2])
        hist_bn, bins_bn = exposure.histogram(image_gray)

        # Layout de la figura
        plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(2, 1)

        # Título de la figura
        plt.suptitle(f"Histogramas de la imagen {i}", y=0.95, fontsize=14)

        # Histograma de la imagen en escala de grises
        ax = plt.subplot(gs[0])
        ax.plot(bins_bn, hist_bn, color="k")
        ax.set_title("Escala de grises", y=1.0, pad=-14, fontsize=10)

        # Histogramas de la imagen para cada canal de color
        ax = plt.subplot(gs[1])
        ax.plot(bins_r, hist_r, color="r")
        ax.plot(bins_g, hist_g, color="g")
        ax.plot(bins_b, hist_b, color="b")
        ax.set_title("RGB", y=1.0, pad=-14, fontsize=10)

    # Mostramos las gráficas
    plt.show()


if __name__ == "__main__":
    main()
