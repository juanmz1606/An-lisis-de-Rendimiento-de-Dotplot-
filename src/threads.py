import argparse
import time
from Bio import SeqIO
from PIL import Image
import numpy as np
from scipy.signal import convolve2d
import threading
from queue import Queue

def dotplot_paralelo(seq1, seq2, rows, cols, results):
    dotplot = [[0 for _ in range(len(seq2))] for _ in range(len(seq1))]

    for i in rows:
        for j in cols:
            if seq1[i] == seq2[j]:
                dotplot[i][j] = 1

    results.put(dotplot)

def dividir_trabajo(num_threads, rows, cols):
    row_segments = np.array_split(np.arange(len(rows)), num_threads)
    col_segments = np.array_split(np.arange(len(cols)), num_threads)
    return row_segments, col_segments

def guardar_dotplot_txt(dotplot, file_output):
    with open(file_output, 'w') as f:
        for fila in dotplot:
            f.write(' '.join(map(str, fila)) + '\n')

def guardar_dotplot_imagen(dotplot, file_output):
    img = Image.new('1', (len(dotplot[0]), len(dotplot)))
    pixels = img.load()

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            color = 255 if dotplot[j][i] == 1 else 0
            pixels[i, j] = color

    img.save(file_output)

def aplicar_convolucion(dotplot, filtro):
    return convolve2d(dotplot, filtro, mode='same', boundary='fill', fillvalue=0).tolist()

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

    # Cargar secuencias desde archivos FASTA
    seq1 = [record.seq[:1000] for record in SeqIO.parse("data/" + args.file1, 'fasta')][0]
    seq2 = [record.seq[:1000] for record in SeqIO.parse("data/" + args.file2, 'fasta')][0]

    # Calcular dotplot
    start_time = time.time()
    num_threads = args.num_processes
    segment_size = 250  # Tamaño de cada segmento

    rows = range(len(seq1))
    cols = range(len(seq2))

    row_segments, col_segments = dividir_trabajo(num_threads, rows, cols)

    threads = []
    dotplots = []
    results_queue = Queue()

    for i in range(num_threads):
        for j in range(num_threads):
            rows_for_thread = row_segments[i]
            cols_for_thread = col_segments[j]

            thread = threading.Thread(target=dotplot_paralelo, args=(seq1, seq2, rows_for_thread, cols_for_thread, results_queue))
            thread.start()
            threads.append(thread)

    for thread in threads:
        thread.join()

    for _ in range(num_threads * num_threads):
        dotplot = results_queue.get()
        dotplots.append(dotplot)

    end_time = time.time()

    print(f"Tiempo de ejecución: {end_time - start_time} segundos")

    # Aplicar convolución
    filtro_diagonal = np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]])

    dotplot = np.zeros((1000, 1000), dtype=int)
    for d in dotplots:
        dotplot += d

    dotplot_diagonal = aplicar_convolucion(dotplot, filtro_diagonal)

    # Guardar dotplot en archivo de texto
    guardar_dotplot_txt(dotplot, args.output_txt_no_f)

    # Guardar dotplot como imagen
    guardar_dotplot_imagen(dotplot, args.output_img_no_f)

    # Guardar dotplot en archivo de texto
    guardar_dotplot_txt(dotplot_diagonal, args.output_txt)

    # Guardar dotplot como imagen
    guardar_dotplot_imagen(dotplot_diagonal, args.output_img)

if __name__ == '__main__':
    main()
