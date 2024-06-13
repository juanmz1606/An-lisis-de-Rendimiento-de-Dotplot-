import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import subprocess
import argparse
import time
import os

def read_csv_file(file_path):
    # Read the CSV data
    data = pd.read_csv(file_path)
    return data

def run_mpi(num_processes, file1, file2, output, output_nf):
    command = [
        "mpirun",
        "-n", str(num_processes),
        "python3", "src/mpi_strategy.py",
        f"--file1={file1}",
        f"--file2={file2}",
        f"--output={output}",
        f"--outputNoFilter={output_nf}"
    ]
    subprocess.run(command)

def run_paralelo(num_processes, file1, file2, output, output_nf):
    command = [
        "python3", "src/threads.py",
        f"-n", str(num_processes),
        f"--file1={file1}",
        f"--file2={file2}",
        f"--output={output}",
        f"--outputNoFilter={output_nf}"
    ]
    subprocess.run(command)

def run_secuencial(file1, file2, output,output_nf, num_seqs=1000):
    command = [
        "python3", "src/secuencial.py",
        f"--file1={file1}",
        f"--file2={file2}",
        f"--output={output}",
        f"--outputNoFilter={output_nf}"
    ]
    subprocess.run(command)

def run_multiprocessing(num_processes, file1, file2, output, output_nf):
    command = [
        "python3", "src/multiprocessing_version.py",
        f"--num_processes={num_processes}",
        f"--file1={file1}",
        f"--file2={file2}",
        f"--output={output}",
        f"--outputNoFilter={output_nf}"
    ]
    res = subprocess.run(command)
    print(res)

def choose_strategy(args, parser):
    if args.estrategia == "mpi":
        if args.num_processes is None:
            parser.error("--num_processes es requerido para la estrategia MPI")
        run_mpi(args.num_processes, args.file1, args.file2, args.output, args.outputNoFilter)
    elif args.estrategia == "secuencial":
        run_secuencial(args.file1, args.file2, args.output, args.outputNoFilter)
    elif args.estrategia == "paralelo":
        run_paralelo(args.num_processes, args.file1, args.file2, args.output, args.outputNoFilter)
    elif args.estrategia == "multiprocessing":
        if args.num_processes is None:
            parser.error("--num_processes es requerido para la estrategia multiprocessing")
        run_multiprocessing(args.num_processes, args.file1, args.file2, args.output, args.outputNoFilter)


def main():
    # parser = argparse.ArgumentParser(description="Script para ejecutar la aplicación")
    # args = parser.parse_args()

    # for nombre_archivo in os.listdir('pruebas'):
    #     os.remove(f"pruebas/{nombre_archivo}")

    # num_processes_list = [2, 4, 8, 10]
    # mpi_list = []
    # secuencial_list = []
    # paralelo_list = []
    # multiprocessing_list = []

    # for strategia in ["multiprocessing", "paralelo", "mpi"]:
    #     args.estrategia = strategia
    #     args.file1 = "E_coli.fna"
    #     args.file2 = "Salmonella.fna"
    #     args.output = "pruebas/pruebas"
    #     args.outputNoFilter = "pruebas/pruebas_nf"
    #     for num_processes in num_processes_list:
    #         args.num_processes = num_processes
    #         start_time = time.time()
    #         choose_strategy(args, parser)
    #         end_time = time.time()
    #         if strategia == "mpi":
    #             mpi_list.append(end_time - start_time)
    #         elif strategia == "paralelo":
    #             paralelo_list.append(end_time - start_time)
    #         elif strategia == "multiprocessing":
    #             multiprocessing_list.append(end_time - start_time)

    # # Tiempo secuencial
    # args.estrategia = "secuencial"
    # args.file1 = "E_coli.fna"
    # args.file2 = "Salmonella.fna"
    # args.output = "pruebas/pruebas"
    # args.outputNoFilter = "pruebas/pruebas_nf"
    # start_time = time.time()
    # choose_strategy(args, parser)
    # end_time = time.time()
    # secuencial_list.append(end_time - start_time)

    # # Lee los archivos csv de una carpeta

    data_secuencial = read_csv_file("files/16000/secuencial.csv")
    data_hilos = read_csv_file("files/16000/hilos.csv")
    data_multi = read_csv_file("files/16000/multi.csv")
    data_mpi = read_csv_file("files/16000/mpi.csv")
    data_pycuda = read_csv_file("files/16000/pycuda.csv")

    # Grafica los datos
    plt.title("Speed up vs Number of Processes")
    tiempo_secuencial = data_secuencial["secuential_time"]
    tiempo_hilos = data_hilos["parallel_time"]
    tiempo_multi = data_multi["parallel_time"]
    tiempo_mpi = data_mpi["parallel_time"]
    tiempo_pycuda = data_pycuda["parallel_time"]


    speed_up_hilos = [tiempo_secuencial / tiempo_hilos[i] for i in range(len(tiempo_hilos))]
    speed_up_multi = [tiempo_secuencial / tiempo_multi[i] for i in range(len(tiempo_multi))]
    speed_up_mpi = [tiempo_secuencial / tiempo_mpi[i] for i in range(len(tiempo_mpi))]
    speed_up_pycuda = [tiempo_secuencial / tiempo_pycuda[i] for i in range(len(tiempo_pycuda))]

    ax = plt.figure(figsize=(10,5)).add_subplot(111)
    ax.plot(data_hilos["num_processes"], speed_up_hilos, linewidth=5, alpha=0.5, label="Threads")
    ax.plot(data_multi["num_processes"], speed_up_multi, linewidth=5, alpha=0.5, label="Multiprocessing")
    ax.plot(data_mpi["num_processes"], speed_up_mpi, linewidth=5, alpha=0.5, label="Mpi")
    ax.plot(data_pycuda["num_processes"], speed_up_pycuda, linewidth=5, alpha=0.5, label="PyCuda")

    ax.set_xlabel("Number of Processes")
    ax.set_ylabel("Speed Up")
    ax.legend()
    plt.savefig("pruebas/speedUp.png")

    efficiency_hilos = [speed_up_hilos[i] / data_hilos["num_processes"][i] for i in range(len(speed_up_hilos))]
    efficiency_multi = [speed_up_multi[i] / data_multi["num_processes"][i] for i in range(len(speed_up_multi))]
    efficiency_mpi = [speed_up_mpi[i] / data_mpi["num_processes"][i] for i in range(len(speed_up_mpi))]
    efficiency_pycuda = [speed_up_pycuda[i] / data_pycuda["num_processes"][i] for i in range(len(speed_up_pycuda))]

    plt.title("Efficiency vs Number of Processes")
    ax = plt.figure(figsize=(10,5)).add_subplot(111)
    ax.plot(data_hilos["num_processes"], efficiency_hilos, linewidth=5, alpha=0.5, label="Threads")
    ax.plot(data_multi["num_processes"], efficiency_multi, linewidth=5, alpha=0.5, label="Multiprocessing")
    ax.plot(data_mpi["num_processes"], efficiency_mpi, linewidth=5, alpha=0.5, label="Mpi")
    ax.plot(data_pycuda["num_processes"], efficiency_pycuda, linewidth=5, alpha=0.5, label="PyCuda")

    ax.set_xlabel("Number of Processes")
    ax.set_ylabel("Efficiency")
    ax.legend()
    plt.savefig("pruebas/efficiency.png")

    data_secuencial = read_csv_file("files/16000/secuencial.csv")
    data_hilos = read_csv_file("files/16000/hilos.csv")
    data_multi = read_csv_file("files/16000/multi.csv")
    data_mpi = read_csv_file("files/16000/mpi.csv")
    data_pycuda = read_csv_file("files/16000/pycuda.csv")

    tiempo_secuencial_t = data_secuencial["secuential_time"]
    tiempo_hilos_t = data_hilos["parallel_time"]
    tiempo_multi_t = data_multi["parallel_time"]
    tiempo_mpi_t = data_mpi["parallel_time"]
    tiempo_pycuda_t = data_pycuda["parallel_time"]

    s_scalability_t = [tiempo_hilos_t[i] for i in range(len(tiempo_hilos_t))]
    s_scalability_mul = [tiempo_multi_t[i] for i in range(len(tiempo_multi_t))]
    s_scalability_mpi = [tiempo_mpi_t[i] for i in range(len(tiempo_mpi_t))]
    s_scalability_pycuda = [tiempo_pycuda_t[i] for i in range(len(tiempo_pycuda_t))]

    s_scalability_t.insert(0, tiempo_secuencial_t[0])
    s_scalability_mul.insert(0, tiempo_secuencial_t[0])
    s_scalability_mpi.insert(0, tiempo_secuencial_t[0])
    s_scalability_pycuda.insert(0, tiempo_secuencial_t[0])


    num_processes = [1,2,4,8]
    plt.title("Strong scalability vs Number of Processes")
    ax = plt.figure(figsize=(10,5)).add_subplot(111)
    ax.plot(num_processes, s_scalability_t, linewidth=5, alpha=0.5, label="Threads")
    ax.plot(num_processes, s_scalability_mul, linewidth=5, alpha=0.5, label="Multiprocessing")
    ax.plot(num_processes, s_scalability_mpi, linewidth=5, alpha=0.5, label="Mpi")
    ax.plot(num_processes, s_scalability_pycuda, linewidth=5, alpha=0.5, label="PyCuda")

    ax.set_xlabel("Number of Processes")
    ax.set_ylabel("Strong Scalability")
    ax.legend()
    plt.savefig("pruebas/strong-scalability.png")


    data_hilos = read_csv_file("files/weak/hilos.csv")
    data_multi = read_csv_file("files/weak/multi.csv")
    data_mpi = read_csv_file("files/weak/mpi.csv")
    data_pycuda = read_csv_file("files/weak/pycuda.csv")

    tiempo_hilos_t = data_hilos["parallel_time"]
    tiempo_multi_t = data_multi["parallel_time"]
    tiempo_mpi_t = data_mpi["parallel_time"]
    tiempo_pycuda_t = data_pycuda["parallel_time"]

    s_scalability_t = [tiempo_hilos_t[i] for i in range(len(tiempo_hilos_t))]
    s_scalability_mul = [tiempo_multi_t[i] for i in range(len(tiempo_multi_t))]
    s_scalability_mpi = [tiempo_mpi_t[i] for i in range(len(tiempo_mpi_t))]
    s_scalability_pycuda = [tiempo_pycuda_t[i] for i in range(len(tiempo_pycuda_t))]

    num_processes = [2,4,8]
    plt.title("Weak scalability vs Number of Processes")
    ax = plt.figure(figsize=(10,5)).add_subplot(111)
    ax.plot(num_processes, s_scalability_t, linewidth=5, alpha=0.5, label="Threads")
    ax.plot(num_processes, s_scalability_mul, linewidth=5, alpha=0.5, label="Multiprocessing")
    ax.plot(num_processes, s_scalability_mpi, linewidth=5, alpha=0.5, label="Mpi")
    ax.plot(num_processes, s_scalability_pycuda, linewidth=5, alpha=0.5, label="PyCuda")


    ax.set_xlabel("Number of Processes")
    ax.set_ylabel("Weak Scalability")
    ax.legend()
    plt.savefig("pruebas/weak-scalability.png")

if __name__ == "__main__":
    main()
