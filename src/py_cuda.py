import time
import numpy as np
from Bio import SeqIO
from PIL import Image
from multiprocessing import Pool
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import argparse
import csv
from scipy.ndimage import convolve
import os

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

def aplicar_filtro_bordes_multiprocessing(imagen, num_procesos):
    pixels = np.array(imagen).astype(np.uint8)
    alto, ancho = pixels.shape

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

def dotplot_pycuda(seq1, seq2):
    dotplot = np.zeros((len(seq1), len(seq2)), dtype=np.int32)
    
    # Convertir secuencias a arrays de numpy
    seq1_array = np.frombuffer(seq1.encode('utf-8'), dtype=np.int8)
    seq2_array = np.frombuffer(seq2.encode('utf-8'), dtype=np.int8)
    
    # Copiar las secuencias a la GPU
    seq1_gpu = cuda.mem_alloc(seq1_array.nbytes)
    seq2_gpu = cuda.mem_alloc(seq2_array.nbytes)
    cuda.memcpy_htod(seq1_gpu, seq1_array)
    cuda.memcpy_htod(seq2_gpu, seq2_array)
    
    # Compilar el kernel de dotplot
    mod = SourceModule("""
    __global__ void dotplot_kernel(int *dotplot, char *seq1, char *seq2, int width) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;
        int index = idy * width + idx;
        
        if (idx < width && idy < width) {
            if (seq1[idx] == seq2[idy]) {
                dotplot[index] = 1;
            }
        }
    }
    """)
    
    dotplot_kernel = mod.get_function("dotplot_kernel")
    
    # Configurar la cuadrícula y los bloques
    block_size = (16, 16, 1)
    grid_size = ((len(seq1) + block_size[0] - 1) // block_size[0], (len(seq2) + block_size[1] - 1) // block_size[1], 1)
    
    # Ejecutar el kernel
    dotplot_kernel(cuda.Out(dotplot), seq1_gpu, seq2_gpu, np.int32(len(seq2)), block=block_size, grid=grid_size)
    
    return dotplot

def guardar_dotplot_txt(dotplot, file_output):
    with open(file_output, 'w') as f:
        for fila in dotplot:
            f.write(' '.join(map(str, fila)) + '\n')

def guardar_dotplot_imagen(dotplot, file_output):
    # Asumiendo que dotplot ya es de tipo np.uint8
    img = Image.fromarray(dotplot.T * 255, 'L')  # 'L' para escala de grises
    img.save(file_output)

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Dotplot paralelo')
    parser.add_argument('-n', '--num_processes', type=int, required=True, help='Número de procesos')
    parser.add_argument('--file1', required=True, help='Archivo FASTA 1')
    parser.add_argument('--file2', required=True, help='Archivo FASTA 2')
    parser.add_argument('-o', '--output', required=True, help='Archivo de salida')
    parser.add_argument('-outnf', '--outputNoFilter', required=True, help='Archivo de salida sin filtro')
    args = parser.parse_args()

    args.output_txt = args.output + ".txt"
    args.output_img = args.output + ".png"
    args.output_txt_no_f = args.outputNoFilter + ".txt"
    args.output_img_no_f = args.outputNoFilter + ".png"
    
    # Medir tiempo de carga de datos
    data_load_start = time.time()
    # Cargar secuencias desde archivos FASTA
    seq1 = str([record.seq[:6000] for record in SeqIO.parse(args.file1, 'fasta')][0])
    seq2 = str([record.seq[:6000] for record in SeqIO.parse(args.file2, 'fasta')][0])
    
    data_load_end = time.time()
    data_load_time = data_load_end - data_load_start
    
    parallel_start = time.time()

    dotplot = dotplot_pycuda(seq1, seq2)
    
    parallel_end = time.time()
    parallel_time = parallel_end - parallel_start
    
    # Medir tiempo de convolución
    convolution_start = time.time()
    
    dotplot_diagonal = aplicar_filtro_bordes_multiprocessing(dotplot, args.num_processes)
    
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
    num_processes = args.num_processes

    end_time = time.time()
    total_time = end_time - start_time
    
    with open(f'/content/drive/My Drive/pycuda/pycuda.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if args.num_processes == 2:
            writer.writerow(['total_time','parallel_time', 'data_load_time', 'convolution_time', 'save_time', 'num_processes'])
        # Escribe los tiempos en el archivo CSV junto con la cantidad de procesos
        writer.writerow([total_time, parallel_time, data_load_time, convolution_time, save_time, args.num_processes])

if __name__ == '__main__':
    main()
