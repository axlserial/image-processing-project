import skimage as ski
import numpy as np
import matplotlib.pyplot as plt

from os import path
from tabulate import tabulate
from utils import read_images


def main():
    # Leemos las imágenes
    images = read_images(__file__, "late_blight")

    # Tablas de datos
    features = []
    texture = []

    for image, file in images:
        # --/ Preprocesamiento / --

        img_float = ski.img_as_float(image)

        # Ecualización de histograma adaptativa
        img_adapteq = ski.exposure.equalize_adapthist(img_float, clip_limit=0.02)

        # --/ Segmentación /--

        # Umbralización Region Adjacency Graph (RAG)
        labels = ski.segmentation.slic(
            img_adapteq, compactness=10, n_segments=100, start_label=1, sigma=4
        )
        rag = ski.color.label2rgb(labels, img_adapteq, kind="avg", bg_label=0)

        # Convertimos la imagen a escala de grises
        rag_gray = ski.color.rgb2gray(rag)

        # Filtro de la mediana
        median_rag = ski.filters.median(rag_gray)

        # Umbralización Otsu
        threshold = ski.filters.threshold_otsu(median_rag)
        otsu_rag = median_rag <= threshold
        otsu_rag = otsu_rag.astype(int)

        # Obtenemos bordes
        edges = ski.segmentation.find_boundaries(otsu_rag, mode="inner")
        edges = edges.astype(int)

        # Operaciones morfológicas para borde
        dilated = ski.morphology.dilation(otsu_rag)
        eroded = ski.morphology.erosion(otsu_rag)
        mask = np.logical_and(dilated, np.logical_not(eroded))

        # Etiquetado de regiones
        labels = ski.measure.label(otsu_rag, background=0)

        # Obtenemos la región de interés
        max_label = np.max(labels)
        img_label = np.where(labels == max_label, 1, 0)
        img_label = img_label.astype(int)
        img_label = otsu_rag

        # Obtenemos el borde de la región de interés
        img_label_edges = ski.segmentation.find_boundaries(img_label, mode="inner")
        img_label_edges = img_label_edges.astype(int)
        img_label_edges = edges

        # --/ Descripción /--

        # Extracción de características
        
        # Obtenemos el área de la imagen, de otsu_filtered_image todos los píxeles que son 1
        m_00 = np.sum(img_label == 1)

        # Obtenemos las coordenadas de los píxeles que son 1
        x, y = np.where(img_label == 1)

        # Obtenemos m_10 y m_01
        m_10 = np.sum(x)
        m_01 = np.sum(y)
        
        x_ = int(m_10 / m_00)
        y_ = int(m_01 / m_00)

        # Calculo del centro de masa
        centro_masa = (x_,y_)

        # Calculamos el perimetro
        perimetro = np.sum(img_label_edges == 1)

        # Calculamos las indices del perimetro
        x_b, y_b = np.where(img_label_edges == 1)

        # Calculamos la distancia radial, con la formula de la distancia euclidiana
        euclidean_distance = np.sqrt((x_b - x_)**2 + (y_b - y_)**2)

        # Calculamos la distancia radial normalizada
        max_distance = np.max(euclidean_distance)
        euclidean_distance_norm = euclidean_distance / max_distance

        # Calculamos la media de la distancia radial normalizada
        mean_eun = np.mean(euclidean_distance_norm)

        # Calculamos el indice de rugosidad
        term = 1 / (perimetro*mean_eun)
        suma = 0
        for i in range(len(euclidean_distance_norm) - 1):
            suma += euclidean_distance_norm[i] - euclidean_distance_norm[i+1]

        rugosity = term * suma

        # Calculamos el indice de area
        suma2 = 0
        for i in range(len(euclidean_distance_norm)):
            suma2 += euclidean_distance_norm[i] - mean_eun

        area_ind = term * suma2

        features.append(
            {
                "Archivo": file,
                "m00": f"{m_00}",
                "m10": f"{m_10}",
                "m01": f"{m_01}",
                "Centro de masa": f"({x_},{y_})",
                "Ind. Area": f"{area_ind}",
                "Ind. Rugosidad": f"{rugosity}",
            }
        )

        # Textura / Estadísticos de primer orden

        # Obtenemos los valores maximos y minimos de los niveles de intensidad
        img_text = ski.util.img_as_ubyte(img_float)

        max_val = np.max(img_text[img_label_edges == 1])
        min_val = np.min(img_text[img_label_edges == 1])
        
        # Calculamos la media
        mean_val = np.mean(img_text[img_label_edges == 1])

        # Calculamos la varianza
        var_val = np.var(img_text[img_label_edges == 1],ddof=1)

        # Calculamos la desviación estándar
        std_val = 1 - (1 / (1 + var_val))

        # Guardamos los datos
        texture.append(
            {
                "Archivo": file,
                "Minimo": f"{min_val}",
                "Máximo": f"{max_val}",
                "Media": f"{mean_val:.4f}",
                "Desv. Estándar": f"{std_val:.4f}",
            }
        )

        # --/ Visualización /--
        plt.figure()
        plt.subplots_adjust(hspace=0.4,wspace=0.5)

        plt.suptitle(
            f"Hoja: {path.basename(file)}",
            y=0.96,
            fontsize=14,
        )

        size_plt = [3, 3]

        # Subplot 1: Original
        plt.subplot(size_plt[0], size_plt[1], 1)
        plt.imshow(image, cmap=plt.cm.gray)
        plt.title("Original")

        # Subplot 2: Ecualización adaptativa
        plt.subplot(size_plt[0], size_plt[1], 2)
        plt.imshow(img_adapteq, cmap=plt.cm.gray)
        plt.title("Ecualización Adaptativa")

        # Subplot 3: RAG
        plt.subplot(size_plt[0], size_plt[1], 3)
        plt.imshow(rag)
        plt.title("RAG")

        # Subplot 4: Otsu RAG
        plt.subplot(size_plt[0], size_plt[1], 4)
        plt.imshow(otsu_rag, cmap=plt.cm.gray)
        plt.title("Otsu RAG")

        # Subplot 5: Bordes -> find_boundaries
        plt.subplot(size_plt[0], size_plt[1], 5)
        plt.imshow(edges, cmap=plt.cm.gray)
        plt.title("Borde: find_boundaries")

        # Subplot 6: Border -> Operaciones morfológicas
        plt.subplot(size_plt[0], size_plt[1], 6)
        plt.imshow(mask, cmap=plt.cm.gray)
        plt.title("Borde: Operaciones morfológicas")

        # Subplot 7: Labels
        plt.subplot(size_plt[0], size_plt[1], 7)
        plt.imshow(labels, cmap=plt.cm.gray)
        plt.title("Labels")

        # Subplot 8: Label
        plt.subplot(size_plt[0], size_plt[1], 8)
        plt.imshow(img_label, cmap=plt.cm.gray)
        plt.title("Region de interés (Label)")

        # Subplot 9: Label edges
        plt.subplot(size_plt[0], size_plt[1], 9)
        plt.imshow(img_label_edges, cmap=plt.cm.gray)
        plt.title("Borde de la región de interés")

    plt.show()

    print('\nExtracción de características: \n')
    print(tabulate(features, headers="keys", tablefmt="grid"))
    print('\n')

    print('\nDescriptores de textura de primer orden: \n')
    print(tabulate(texture, headers="keys", tablefmt="grid"))
    print('\n\n')


if __name__ == "__main__":
    main()
