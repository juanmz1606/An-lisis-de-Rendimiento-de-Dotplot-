import argparse
import time
from Bio import SeqIO
from PIL import Image
import numpy as np
from scipy.signal import convolve2d
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

def aplicar_convolucion(dotplot, filtro):
    return convolve2d(dotplot, filtro, mode='same', boundary='fill', fillvalue=0)

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
    
    end_time = time.time()
    print(f"Tiempo de ejecución: {end_time - start_time} segundos")

    # Aplicar convolución
    filtro_diagonal = np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]])

    dotplot_diagonal = aplicar_convolucion(dotplot, filtro_diagonal)

    # Guardar dotplot en archivo de texto
    guardar_dotplot_txt(dotplot, args.output_txt_no_f)
    guardar_dotplot_txt(dotplot_diagonal, args.output_txt)

    # Guardar dotplot como imagen
    guardar_dotplot_imagen(dotplot, args.output_img_no_f)
    guardar_dotplot_imagen(dotplot_diagonal, args.output_img)

if __name__ == '__main__':
    main()
