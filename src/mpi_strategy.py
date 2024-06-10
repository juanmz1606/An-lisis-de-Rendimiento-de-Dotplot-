from mpi4py import MPI
import argparse
import time
from Bio import SeqIO
from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def dotplot_mpi(seq1, seq2, start, end):
    dotplot = [[0 for _ in range(len(seq2))] for _ in range(end - start)]

    for i in range(start, end):
        for j in range(len(seq2)):
            if seq1[i] == seq2[j]:
                dotplot[i - start][j] = 1

    return (start, end), dotplot

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

def aplicar_convolucion(dotplot, filtro):
    return convolve2d(dotplot, filtro, mode='same', boundary='fill', fillvalue=0).tolist()

def main():
    # Inicializar MPI
    comm = MPI.COMM_WORLD
    # Obtiene el id del proceso
    rank = comm.Get_rank()
    # Obtiene el número de procesos
    size = comm.Get_size()

    parser = argparse.ArgumentParser(description='Dotplot paralelo')
    parser.add_argument('--file1', required=True, help='Archivo FASTA 1')
    parser.add_argument('--file2', required=True, help='Archivo FASTA 2')
    parser.add_argument('--num_seqs', type=int, default=100, help='Número de secuencias a tomar de cada archivo FASTA')
    parser.add_argument('--output', required=True, help='Archivo de salida')
    parser.add_argument('--outputNoFilter', required=True, help='Archivo de salida sin filtro')

    args = parser.parse_args()

    args.output_txt = args.output + ".txt"
    args.output_img = args.output + ".png"
    args.output_txt_no_f = args.outputNoFilter + ".txt"
    args.output_img_no_f = args.outputNoFilter + ".png"

    # Cargar secuencias desde archivos FASTA
    if rank == 0:
        print(f"Procesando archivos {args.file1} y {args.file2} con {size} procesos")
        seq1 = [record.seq[:1000] for record in SeqIO.parse("data/" + args.file1, 'fasta')][0]
        seq2 = [record.seq[:1000] for record in SeqIO.parse("data/" + args.file2, 'fasta')][0]

        # Verificar que las secuencias sean del mismo tamaño
        if len(seq1) != len(seq2):
            print("Las secuencias deben ser del mismo tamaño")
            return

        # Broadcast de las secuencias a todos los procesos
        seq1 = comm.bcast(seq1, root=0)
        seq2 = comm.bcast(seq2, root=0)
    else:
        # Recibir secuencias
        seq1 = comm.bcast(None, root=0)
        seq2 = comm.bcast(None, root=0)

    # Dividir las secuencias entre los procesos
    start = rank * (len(seq1) // size) + min(rank, len(seq1) % size)
    end = (rank + 1) * (len(seq1) // size) + min(rank + 1, len(seq1) % size)

    # Calcular dotplot localmente
    start_time = time.time()
    dotplot_rango, dotplot_parcial = dotplot_mpi(seq1, seq2, start, end)
    end_time = time.time()

    # Realizar un gather de los resultados
    dotplot_local = comm.gather((dotplot_rango, dotplot_parcial), root=0)

    print(f"Tiempo de ejecución en proceso {rank}: {end_time - start_time} segundos")

    # Reunir los dotplots en el proceso maestro
    if rank == 0:
        dotplot = [0] * len(seq1)
        for rango, dotplot_parcial in dotplot_local:
            start, end = rango
            for i in range(start, end):
                dotplot[i] = dotplot_parcial[i - start]

        # Aplicar convolución
        filtro_diagonal = np.array([[1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1]])
        dotplot_diagonal = aplicar_convolucion(dotplot, filtro_diagonal)

        # Guardar dotplot sin filtro en archivo de texto
        guardar_dotplot_txt(dotplot, args.output_txt_no_f)
        # Guardar dotplot sin filtro como imagen
        guardar_dotplot_imagen(dotplot, args.output_img_no_f)

        # Guardar dotplot con filtro en archivo de texto
        guardar_dotplot_txt(dotplot_diagonal, args.output_txt)
        # Guardar dotplot con filtro como imagen
        guardar_dotplot_imagen(dotplot_diagonal, args.output_img)

        print("Dotplot guardado en", args.output_txt)

if __name__ == '__main__':
    main()
