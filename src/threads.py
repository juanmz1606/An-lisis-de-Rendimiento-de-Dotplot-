import argparse
import time
from Bio import SeqIO
from PIL import Image
import numpy as np
from multiprocessing import Pool
import csv
from scipy.ndimage import convolve
import os
from concurrent.futures import ThreadPoolExecutor

def aplicar_filtro_seccion(args):
    pixels, inicio, fin, ancho = args
    # Kernel de Sobel para bordes verticales
    kernel_y = np.array([
        [0.4, -0.001, 0.4],
        [-0.001, 0.4, -0.001],
        [0.4, -0.001, 0.4]
    ], dtype=np.float32)

    # Sección con espacio para superposición
    seccion_ampliada = pixels[max(inicio-1, 0):fin+1, :]

    # Aplicar convolución usando el kernel
    bordes_seccion = convolve(seccion_ampliada, kernel_y, mode='constant', cval=0.0)

    # Recortar los bordes para mantener el resultado dentro de los límites originales
    bordes_seccion = np.clip(bordes_seccion, 0, 255).astype(np.uint8)
    bordes_seccion = bordes_seccion[inicio > 0:-1 if fin != pixels.shape[0] else None, :]

    return bordes_seccion


# Esta función se encarga de gestionar los procesos y los "pedazos" de imagen que le manda a cada proceso.

def aplicar_filtro_bordes_multiprocessing(imagen):
    pixels = np.array(imagen).astype(np.uint8)
    alto, ancho = pixels.shape
    num_procesos = os.cpu_count()  # Ajustar basado en el número de núcleos de CPU

    # Dividir la imagen en secciones con superposición adecuada
    seccion_alto = alto // num_procesos
    secciones = [(pixels, max(i * seccion_alto - 3, 0), min((i + 1) * seccion_alto + 3, alto), ancho)
                 for i in range(num_procesos)]

    # Crear un pool de procesos y aplicar el filtro a cada sección
    with Pool(num_procesos) as pool:
        resultados = pool.map(aplicar_filtro_seccion, secciones)

    # Combinar los resultados con cuidado para evitar duplicar las áreas de superposición
    bordes = np.vstack(resultados).astype(np.uint8)

    return bordes

def calcular_dotplot_seccion(args):
    seq1_array, seq2_array, start, end = args
    dotplot_section = (seq1_array[:, None] == seq2_array[start:end]).astype(np.uint8)
    return dotplot_section

def dotplot_paralelo(seq1, seq2, num_processes):
    seq1_array = np.array(list(seq1))
    seq2_array = np.array(list(seq2))

    seccion_ancho = len(seq2_array) // num_processes
    secciones = [(seq1_array, seq2_array, i * seccion_ancho, (i + 1) * seccion_ancho if i < num_processes - 1 else len(seq2_array))
                 for i in range(num_processes)]

    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        resultados = list(executor.map(calcular_dotplot_seccion, secciones))

    dotplot = np.hstack(resultados).astype(np.uint8)
    return dotplot

# def guardar_dotplot_txt(dotplot, file_output):
#     with open(file_output, 'w') as f:
#         for fila in dotplot:
#             f.write(' '.join(map(str, fila)) + '\n')


def guardar_dotplot_imagen(dotplot, file_output):
    # Asumiendo que dotplot ya es de tipo np.uint8
    img = Image.fromarray(dotplot.T * 255, 'L')  # 'L' para escala de grises
    img.save(file_output)


def main():
    parser = argparse.ArgumentParser(description='Dotplot paralelo')
    parser.add_argument("-n", "--num_processes", type=int, required=True, help="Número de procesos")
    parser.add_argument('--file1', required=True, help='Archivo FASTA 1')
    parser.add_argument('--file2', required=True, help='Archivo FASTA 2')
    parser.add_argument('--output', required=True, help='Archivo de salida')
    parser.add_argument('--outputNoFilter', required=True, help='Archivo de salida sin filtro')

    args = parser.parse_args()

    args.output_txt = args.output + ".txt"
    args.output_img = args.output + ".png"
    args.output_txt_no_f = args.outputNoFilter + ".txt"
    args.output_img_no_f = args.outputNoFilter + ".png"

    # Medir tiempo de carga de datos
    start_time = time.time()
    data_load_start = start_time

    # Cargar secuencias desde archivos FASTA
    seq1 = [record.seq[:6000] for record in SeqIO.parse("data/" + args.file1, 'fasta')][0]
    seq2 = [record.seq[:6000] for record in SeqIO.parse("data/" + args.file2, 'fasta')][0]

    data_load_end = time.time()
    data_load_time = data_load_end - data_load_start

    # Calcular dotplot
    parallel_start = time.time()
    dotplot = dotplot_paralelo(seq1, seq2, args.num_processes)
    parallel_end = time.time()
    parallel_time = parallel_end - parallel_start

    # Medir tiempo de convolución
    convolution_start = time.time()

    dotplot_diagonal = aplicar_filtro_bordes_multiprocessing(dotplot)

    convolution_end = time.time()
    convolution_time = convolution_end - convolution_start

    # Guardar dotplot en archivo de texto
    save_start = time.time()

    # Guardar dotplot en archivo de texto
    #guardar_dotplot_txt(dotplot, args.output_txt_no_f)

    # Guardar dotplot como imagen
    #guardar_dotplot_imagen(dotplot, args.output_img_no_f)

    # Guardar dotplot en archivo de texto
    #guardar_dotplot_txt(dotplot_diagonal, args.output_txt)

    # Guardar dotplot como imagen
    guardar_dotplot_imagen(dotplot_diagonal, args.output_img)

    save_end = time.time()
    save_time = save_end - save_start

    end_time = time.time()
    total_time = end_time - start_time

    # Abre un archivo CSV en modo escritura
    with open('pruebas/hilos.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if args.num_processes == 2:
            # Escribe los tiempos en el archivo CSV
            writer.writerow(['total_time', 'parallel_time', 'data_load_time', 'convolution_time', 'save_time'])
        writer.writerow([total_time, parallel_time, data_load_time, convolution_time, save_time])


if __name__ == '__main__':
    main()
