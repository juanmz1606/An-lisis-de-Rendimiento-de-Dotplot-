import argparse
import time
from Bio import SeqIO
from PIL import Image
import numpy as np
from multiprocessing import Pool

def aplicar_filtro_seccion(args):
    pixels, inicio, fin, ancho = args
    # Kernel de Sobel para bordes verticales
    kernel_y = np.array([
        [0.4, -0.001, 0.4],
        [-0.001, 0.4, -0.001],
        [0.4, -0.001, 0.4]
    ])

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


# Esta función se encarga de gestionar los procesos y los "pedazos" de imagen que le manda a cada proceso.

def aplicar_filtro_bordes_multiprocessing(imagen):
    pixels = np.array(imagen)
    alto, ancho = pixels.shape
    num_procesos = 1

    # Dividir la imagen en secciones con superposición
    seccion_alto = alto // num_procesos
    secciones = []
    for i in range(num_procesos):
        inicio = i * seccion_alto
        fin = (i + 1) * seccion_alto if i != num_procesos - 1 else alto
        if i == 0:
            fin += 3
        if i != 0:
            inicio -= 3  # Superposición para cubrir bordes
        secciones.append((pixels, inicio, fin, ancho))

    # Crear un pool de procesos y aplicar el filtro a cada sección
    with Pool() as pool:
        resultados = pool.map(aplicar_filtro_seccion, secciones)

    # Combinar los resultados
    bordes = np.vstack(resultados)

    return bordes


def dotplot_secuencial(seq1, seq2):
    dotplot = [[0 for _ in range(len(seq2))] for _ in range(len(seq1))]

    for i in range(len(seq1)):
        for j in range(len(seq2)):
            if seq1[i] == seq2[j]:
                dotplot[i][j] = 1

    return dotplot

def guardar_dotplot_txt(dotplot, file_output):

    with open(file_output, 'w') as f:
        for fila in dotplot:
            f.write(' '.join(map(str, fila)) + '\n')

def guardar_dotplot_imagen(dotplot, file_output):
    img = Image.new('1', (len(dotplot[0]), len(dotplot)))
    pixels = img.load()

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixels[i, j] = int(dotplot[j][i])

    img.save(file_output)

def main():
    parser = argparse.ArgumentParser(description='Dotplot secuencial')
    parser.add_argument('--file1', required=True, help='Archivo FASTA 1')
    parser.add_argument('--file2', required=True, help='Archivo FASTA 2')
    parser.add_argument('--output', required=True, help='Archivo de salida')
    parser.add_argument('--outputNoFilter', required=True, help='Archivo de salida sin filtro')

    args = parser.parse_args()

    args.output_txt = args.output + ".txt"
    args.output_img = args.output + ".png"
    args.output_txt_no_f = args.outputNoFilter + ".txt"
    args.output_img_no_f = args.outputNoFilter + ".png"

    # Cargar secuencias desde archivos FASTA
    seq1 = [record.seq[:1000] for record in SeqIO.parse("data/" + args.file1, 'fasta')][0]
    seq2 = [record.seq[:1000] for record in SeqIO.parse("data/" + args.file2, 'fasta')][0]

    # Calcular dotplot
    start_time = time.time()
    dotplot = dotplot_secuencial(seq1, seq2)
    end_time = time.time()

    print(f"Tiempo de ejecución: {end_time - start_time} segundos")
        
    # Guardar dotplot en archivo de texto
    guardar_dotplot_txt(dotplot, args.output_txt_no_f)

    # Guardar dotplot como imagen
    guardar_dotplot_imagen(dotplot, args.output_img_no_f)

    dotplot_diagonal = aplicar_filtro_bordes_multiprocessing(dotplot)

    # Guardar dotplot en archivo de texto
    guardar_dotplot_txt(dotplot_diagonal, args.output_txt)

    # Guardar dotplot como imagen
    guardar_dotplot_imagen(dotplot_diagonal, args.output_img)

if __name__ == '__main__':
    main()
