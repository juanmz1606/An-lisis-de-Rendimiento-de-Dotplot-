from mpi4py import MPI
import argparse
import time
from Bio import SeqIO
from PIL import Image
import numpy as np
from multiprocessing import Pool
import csv

def aplicar_filtro_seccion(args):
    pixels, inicio, fin, ancho = args
    # Kernel de Sobel para bordes verticales
    kernel_y = np.array([
        [0.4, -0.001, 0.4],
        [-0.001, 0.5, -0.001],
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

def aplicar_filtro_bordes_multiprocessing(imagen, size):
    pixels = np.array(imagen)
    alto, ancho = pixels.shape
    num_procesos = size

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
    with Pool(processes=size) as pool:
        resultados = pool.map(aplicar_filtro_seccion, secciones)

    # Combinar los resultados
    bordes = np.vstack(resultados)

    return Image.fromarray(bordes)

def dotplot_mpi(seq1, seq2, start, end):
    dotplot = [[0 for _ in range(len(seq2))] for _ in range(end - start)]

    for i in range(start, end):
        for j in range(len(seq2)):
            if seq1[i] == seq2[j]:
                dotplot[i - start][j] = 1

    return (start, end), dotplot

def guardar_dotplot_txt(dotplot, file_output):
    dotplot = np.array(dotplot)
    with open(file_output, 'w') as f:
        for fila in dotplot:
            f.write(' '.join(map(str, fila)) + '\n')

def guardar_dotplot_imagen(dotplot, file_output):
    dotplot = np.array(dotplot)
    img = Image.new('1', (len(dotplot[0]), len(dotplot)))
    pixels = img.load()

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixels[i, j] = int(dotplot[j][i])

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

        seq1 = [record.seq[:2600] for record in SeqIO.parse("data/" + args.file1, 'fasta')][0]
        seq2 = [record.seq[:2600] for record in SeqIO.parse("data/" + args.file2, 'fasta')][0]

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

        end_time = time.time()
        parallel_time = end_time - start_time

        convolution_start = time.time()

        dotplot_diagonal = aplicar_filtro_bordes_multiprocessing(dotplot, size)

        convolution_end = time.time()
        convolution_time = convolution_end - convolution_start


        save_start = time.time()
        # Guardar dotplot sin filtro en archivo de texto
        guardar_dotplot_txt(dotplot, args.output_txt_no_f)
        # Guardar dotplot sin filtro como imagen
        guardar_dotplot_imagen(dotplot, args.output_img_no_f)

        # Guardar dotplot con filtro en archivo de texto
        guardar_dotplot_txt(dotplot_diagonal, args.output_txt)
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
