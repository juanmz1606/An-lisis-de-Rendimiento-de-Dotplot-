import argparse
import time
from Bio import SeqIO
from PIL import Image
import numpy as np
import multiprocessing
from multiprocessing import Pool
import csv


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

def aplicar_filtro_bordes_multiprocessing(imagen,num_processes):
    pixels = np.array(imagen)
    alto, ancho = pixels.shape
    num_procesos = num_processes

    # Dividir la imagen en secciones con superposición
    seccion_alto = alto // num_procesos
    secciones = []
    for i in range(num_procesos):
        inicio = i * seccion_alto
        fin = (i + 1) * seccion_alto if i != num_procesos - 1 else alto
        if i ==0:
          fin += 3
        if i != 0:
            inicio -= 3  # Superposición para cubrir bordes
        secciones.append((pixels, inicio, fin, ancho))

    # Crear un pool de procesos y aplicar el filtro a cada sección
    with Pool(processes=num_procesos) as pool:
        resultados = pool.map(aplicar_filtro_seccion, secciones)

    # Combinar los resultados
    bordes = np.vstack(resultados)

    return Image.fromarray(bordes)

def dotplot_multiprocessing(args):
    seq1, seq2, rows, cols = args
    dotplot = np.zeros((len(seq1), len(seq2)), dtype=int)

    for i in rows:
        for j in cols:
            if seq1[i] == seq2[j]:
                dotplot[i][j] = 1

    return dotplot

def dividir_trabajo(num_processes, length):
    segments = np.array_split(np.arange(length), num_processes)
    return segments

def guardar_dotplot_txt(dotplot, file_output):
    np.savetxt(file_output, dotplot, fmt='%d')

def guardar_dotplot_imagen(dotplot, file_output):
    dotplot = np.array(dotplot)
    img = Image.new('1', (len(dotplot[0]), len(dotplot)))
    pixels = img.load()

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixels[i, j] = int(dotplot[j][i])

    img.save(file_output)


def main():

    start_time = time.time()

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
    data_load_start = time.time()

    # Cargar secuencias desde archivos FASTA
    seq1 = [record.seq[:2600] for record in SeqIO.parse("data/" + args.file1, 'fasta')][0]
    seq2 = [record.seq[:2600] for record in SeqIO.parse("data/" + args.file2, 'fasta')][0]

    data_load_end = time.time()
    data_load_time = data_load_end - data_load_start

    # Calcular dotplot
    parallel_start = time.time()
    num_processes = args.num_processes

    rows = range(len(seq1))
    cols = range(len(seq2))

    row_segments = dividir_trabajo(num_processes, len(rows))
    col_segments = dividir_trabajo(num_processes, len(cols))

    tasks = [(seq1, seq2, row_segment, col_segment)
             for row_segment in row_segments
             for col_segment in col_segments]

    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(dotplot_multiprocessing, tasks)

    dotplot = np.sum(results, axis=0)

    parallel_end = time.time()
    parallel_time = parallel_end - parallel_start

    # Medir tiempo de convolución
    convolution_start = time.time()
    dotplot_diagonal = aplicar_filtro_bordes_multiprocessing(dotplot,args.num_processes)
    convolution_end = time.time()
    convolution_time = convolution_end - convolution_start

    # Guardar dotplot en archivo de texto
    save_start = time.time()

    guardar_dotplot_txt(dotplot, args.output_txt_no_f)
    guardar_dotplot_txt(dotplot_diagonal, args.output_txt)

    # Guardar dotplot como imagen
    guardar_dotplot_imagen(dotplot, args.output_img_no_f)
    guardar_dotplot_imagen(dotplot_diagonal, args.output_img)

    save_end = time.time()
    save_time = save_end - save_start

    # Calcular métricas
    T1 = parallel_time
    num_processes = args.num_processes

    end_time = time.time()
    total_time = end_time - start_time

    with open(f'pruebas/multi.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if num_processes == 2:
            writer.writerow(['total_time','parallel_time', 'data_load_time', 'convolution_time', 'save_time', 'num_processes'])
        # Escribe los tiempos en el archivo CSV junto con la cantidad de procesos
        writer.writerow([total_time,parallel_time, data_load_time, convolution_time, save_time, num_processes])


if __name__ == '__main__':
    main()
