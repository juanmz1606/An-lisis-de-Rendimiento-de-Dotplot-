import argparse
import time
from Bio import SeqIO
from PIL import Image
import numpy as np
from scipy.ndimage import convolve
import multiprocessing

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
    img = Image.fromarray(np.uint8(dotplot * 255), 'L')
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
    seq1 = [record.seq[:1000] for record in SeqIO.parse("data/" + args.file1, 'fasta')][0]
    seq2 = [record.seq[:1000] for record in SeqIO.parse("data/" + args.file2, 'fasta')][0]

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

    # Definir un filtro personalizado
    filter_matrix = np.array([
        [-0.5, 1, -0.5],
        [1, 3, 1],
        [-0.5, 1, -0.5]
    ])

    dotplot_diagonal = convolve(dotplot, filter_matrix, mode='constant', cval=0.0)
    
    # Ajustar los valores fuera de la diagonal principal
    for i in range(dotplot_diagonal.shape[0]):
        for j in range(dotplot_diagonal.shape[1]):
            if i != j:
                dotplot_diagonal[i][j] *= 0.5  # Atenuar los valores fuera de la diagonal

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

    end_time = time.time()
    total_time = end_time - start_time

    # Calcular métricas
    T1 = parallel_time
    Tp = total_time
    num_processes = args.num_processes
    speedup = T1 / Tp
    efficiency = speedup / num_processes

    # Imprimir resultados
    print(f"Tiempo de carga de datos: {data_load_time} segundos")
    print(f"Tiempo de ejecución total: {total_time} segundos")
    print(f"Tiempo de ejecución paralelizable: {parallel_time} segundos")
    print(f"Tiempo de convolución: {convolution_time} segundos")
    print(f"Tiempo de guardado de datos: {save_time} segundos")
    print(f"Tiempo muerto: {total_time - (parallel_time + data_load_time + save_time + convolution_time)} segundos")
    print(f"Aceleración (Speedup): {speedup}")
    print(f"Eficiencia: {efficiency}")

if __name__ == '__main__':
    main()
