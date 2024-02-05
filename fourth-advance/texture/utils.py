from os import listdir, path
import skimage as ski
import numpy as np


def read_images(actual_file: str, folder: str):
    """Función que lee las imágenes de la carpeta especificada.

    Parameters
    ----------
    actual_file : str
        Nombre del archivo actual.
    folder : str
        Nombre de la carpeta que contiene las imágenes.

    Yields
    ------
    img : numpy.ndarray
        Imagen leída.
    """

    # Obtenemos la ruta absoluta del archivo actual
    current_path = path.dirname(path.realpath(actual_file))
    while current_path.endswith("image-processing-project") is False:
        current_path = path.dirname(current_path)

    # Obtenemos la ruta de la carpeta de imágenes
    img_path = path.join(current_path, "images", folder)

    for file in listdir(img_path):
        img = ski.io.imread(f"{img_path}/{file}")
        img_data: tuple[np.ndarray, str] = (img, file)

        yield img_data
