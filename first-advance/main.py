from os import listdir, path
from skimage import io, exposure, color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import basename


def read_images(path: str):
    """Función que lee las imágenes de un directorio y las devuelve una a una"""

    for file in listdir(path):
        img = io.imread(f"{path}/{file}")
        yield img, file


def main():
    # Obtenemos la ruta absoluta del archivo actual
    dir_path = path.dirname(path.realpath(__file__))

    # Leemos las imágenes
    images = read_images(path.join(dir_path, "images"))

    # Mostramos la ecualización de histograma de las imágenes
    for image, file in images:
        # Convertimos la imagen a escala de grises
        image_gray = color.rgb2gray(image)

        # Obtiene los histogramas para cada canal de color
        hist_r, bins_r = exposure.histogram(image[:, :, 0])
        hist_g, bins_g = exposure.histogram(image[:, :, 1])
        hist_b, bins_b = exposure.histogram(image[:, :, 2])
        hist_bn, bins_bn = exposure.histogram(image_gray)

        # Layout de la figura
        plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(3, 2, hspace=0.3)

        # Título de la figura
        #plt.suptitle(f"Histogramas y ecualizado de la imagen {i}", y=0.95, fontsize=14)
        plt.suptitle(f"Histogramas y ecualizado de la imagen {path.basename(file)}", y=0.95, fontsize=12)

        # Histograma de la imagen en escala de grises
        ax = plt.subplot(gs[0, 0])
        ax.plot(bins_bn, hist_bn, color="k")
        ax.set_title("Escala de grises", y=1.0, pad=-14, fontsize=10)

        # Histogramas de la imagen para cada canal de color
        ax = plt.subplot(gs[0, 1])
        ax.plot(bins_r, hist_r, color="r")
        ax.plot(bins_g, hist_g, color="g")
        ax.plot(bins_b, hist_b, color="b")
        ax.set_title("RGB", y=1.0, pad=-14, fontsize=10)

        # Ecualización de histograma de la imagen en escala de grises
        
        # --/ Mostramos la imagen original
        ax = plt.subplot(gs[1, 0])
        ax.imshow(image_gray, cmap="gray")
        ax.set_xlabel("Original", fontsize=10)

        # --/ Mostramos la imagen ecualizada
        ax = plt.subplot(gs[2, 0])
        ax.imshow(exposure.equalize_hist(image_gray), cmap="gray")
        ax.set_xlabel("Ecualizada", fontsize=10)

        # Ecualización de histograma de la imagen en RGB

        # --/ Mostramos la imagen original en RGB
        ax = plt.subplot(gs[1, 1])
        ax.imshow(image)
        ax.set_xlabel("Original", fontsize=10)

        img_r = exposure.equalize_hist(image[:,:,0])
        img_g = exposure.equalize_hist(image[:,:,1])
        img_b = exposure.equalize_hist(image[:,:,2])

        # --/ Mostramos la imagen ecualizada
        ax = plt.subplot(gs[2, 1])
        ax.imshow(np.concatenate((img_r[..., np.newaxis], img_g[..., np.newaxis], img_b[..., np.newaxis]), axis=-1))
        ax.set_xlabel("Ecualizada", fontsize=10)

    # Mostramos las gráficas
    plt.show()


if __name__ == "__main__":
    main()
