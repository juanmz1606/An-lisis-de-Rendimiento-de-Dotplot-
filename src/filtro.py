from mpi4py import MPI
import numpy as np
from PIL import Image

def aplicar_filtro_seccion(pixels, inicio, fin, ancho):
    # Kernel de Sobel para bordes verticales
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Sección con espacio para superposición
    seccion_ampliada = pixels[inicio-1:fin+1, :] if inicio > 0 else pixels[inicio:fin+1, :]
    bordes_seccion = np.zeros_like(seccion_ampliada, dtype=np.uint8)

    for i in range(1, seccion_ampliada.shape[0] - 1):
        for j in range(1, ancho - 1):
            gy = np.sum(np.multiply(seccion_ampliada[i-1:i+2, j-1:j+2], kernel_y))
            bordes_seccion[i, j] = min(255, np.abs(gy))

    # Eliminar la fila adicional al final si no es la última sección
    if fin != pixels.shape[0]:
        bordes_seccion = bordes_seccion[:-1, :]
    # Eliminar la primera fila si no es la primera sección
    if inicio > 0:
        bordes_seccion = bordes_seccion[1:, :]

    return bordes_seccion

def aplicar_filtro_bordes_mpi4py(imagen):
    # Inicializar MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        pixels = np.array(imagen)
        alto, ancho = pixels.shape
    else:
        pixels = None
        alto = None
        ancho = None

    # Broadcast dimensiones de la imagen
    alto = comm.bcast(alto, root=0)
    ancho = comm.bcast(ancho, root=0)

    # Dividir la imagen en secciones con superposición
    seccion_alto = alto // size
    inicio = rank * seccion_alto
    fin = (rank + 1) * seccion_alto if rank != size - 1 else alto
    if rank == 0:
        fin += 3
    if rank != 0:
        inicio -= 3  # Superposición para cubrir bordes

    # Scatter la imagen
    local_pixels = np.zeros((fin - inicio + 2, ancho), dtype=np.uint8)
    comm.Scatterv([pixels, (fin - inicio + 2) * ancho, None, MPI.UNSIGNED_CHAR], local_pixels, root=0)

    # Aplicar el filtro a la sección local
    local_bordes = aplicar_filtro_seccion(local_pixels, 1, local_pixels.shape[0] - 1, ancho)

    # Gather los resultados
    if rank == 0:
        bordes = np.zeros_like(pixels, dtype=np.uint8)
    else:
        bordes = None

    comm.Gatherv(local_bordes, [bordes, (fin - inicio) * ancho, None, MPI.UNSIGNED_CHAR], root=0)

    if rank == 0:
        return Image.fromarray(bordes)
    else:
        return None

# Ejemplo de uso:
if __name__ == "__main__":
    resultado = aplicar_filtro_bordes_mpi4py(imagen)
    if resultado is not None:
        resultado.save("imagen_procesada.png")
