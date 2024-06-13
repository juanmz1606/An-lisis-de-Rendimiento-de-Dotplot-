from scipy.ndimage import convolve
from multiprocessing import Pool
from mpi4py import MPI
from Bio import SeqIO
from PIL import Image
import numpy as np
import argparse
import time
import csv
import os


def aplicar_filtro_seccion(args):
    pixels, inicio, fin, ancho = args
    kernel_y = np.array([
        [0.4, -0.001, 0.4],
        [-0.001, 0.5, -0.001],
        [0.4, -0.001, 0.4]
    ]).astype(np.float32)

    seccion_ampliada = pixels[inicio-1:fin+1, :] if inicio > 0 else pixels[inicio:fin+1, :]
    bordes_seccion = convolve(seccion_ampliada, kernel_y)
    bordes_seccion = np.clip(np.abs(bordes_seccion), 0, 255)

    if fin != pixels.shape[0]:
        bordes_seccion = bordes_seccion[:-1, :]
    if inicio > 0:
        bordes_seccion = bordes_seccion[1:, :]

    return bordes_seccion.astype(np.uint8)

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

def dotplot_mpi(seq1, seq2, start, end):
    seq1_slice = seq1[start:end]
    dotplot = np.array(np.array(seq1_slice)[:, None] == np.array(seq2)[None, :], dtype=np.uint8)
    return (start, end), dotplot

def guardar_dotplot_txt(dotplot, file_output):
    with open(file_output, 'w') as f:
        for fila in dotplot:
            f.write(' '.join(map(str, fila)) + '\n')

def guardar_dotplot_imagen(dotplot, file_output):
    img = Image.fromarray(dotplot.T * 255, 'L')  # 'L' para escala de grises
    img.save(file_output)

def main():
    # Inicializar MPI
    comm = MPI.COMM_WORLD
    # Obtiene el id del proceso
    rank = comm.Get_rank()
    # Obtiene el número de procesos
    size = comm.Get_size()

    if rank == 0:
        start_time = time.time()

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
        # print(f"Procesando archivos {args.file1} y {args.file2} con {size} procesos")
        data_load_start = time.time()

        seq1 = [record.seq[:16000] for record in SeqIO.parse("data/" + args.file1, 'fasta')][0]
        seq2 = [record.seq[:16000] for record in SeqIO.parse("data/" + args.file2, 'fasta')][0]

        data_load_end = time.time()
        data_load_time = data_load_end - data_load_start

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
    # start_time = time.time()
    dotplot_rango, dotplot_parcial = dotplot_mpi(seq1, seq2, start, end)
    # end_time = time.time()

    # Realizar un gather de los resultados
    dotplot_local = comm.gather((dotplot_rango, dotplot_parcial), root=0)

    # print(f"Tiempo de ejecución en proceso {rank}: {end_time - start_time} segundos")

    # Reunir los dotplots en el proceso maestro
    if rank == 0:
        dotplot = [0] * len(seq1)
        for rango, dotplot_parcial in dotplot_local:
            start, end = rango
            for i in range(start, end):
                dotplot[i] = dotplot_parcial[i - start]

        dotplot = np.array(dotplot).astype(np.uint8)

        end_time = time.time()
        parallel_time = end_time - start_time

        convolution_start = time.time()

        dotplot_diagonal = aplicar_filtro_bordes_multiprocessing(dotplot)

        convolution_end = time.time()
        convolution_time = convolution_end - convolution_start


        save_start = time.time()
        # Guardar dotplot sin filtro en archivo de texto
        # guardar_dotplot_txt(dotplot, args.output_txt_no_f)
        # Guardar dotplot sin filtro como imagen
        # guardar_dotplot_imagen(dotplot, args.output_img_no_f)

        # Guardar dotplot con filtro en archivo de texto
        # guardar_dotplot_txt(dotplot_diagonal, args.output_txt)
        # Guardar dotplot con filtro como imagen
        guardar_dotplot_imagen(dotplot_diagonal, args.output_img)

        save_end = time.time()
        save_time = save_end - save_start

        end_time = time.time()
        total_time = end_time - start_time

        with open(f'pruebas/mpi.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            if size == 2:
                writer.writerow(['total_time','parallel_time', 'data_load_time', 'convolution_time', 'save_time', 'num_processes'])
            # Escribe los tiempos en el archivo CSV junto con la cantidad de procesos
            writer.writerow([total_time, parallel_time, data_load_time, convolution_time, save_time, size])


if __name__ == '__main__':
    main()
