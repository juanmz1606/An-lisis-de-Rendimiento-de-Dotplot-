import subprocess
import argparse
import time
import os

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
    parser = argparse.ArgumentParser(description="Script para ejecutar la aplicación")
    args = parser.parse_args()

    for nombre_archivo in os.listdir('pruebas'):
        os.remove(f"pruebas/{nombre_archivo}")

    num_processes_list = [2, 4, 8]
    mpi_list = []
    secuencial_list = []
    paralelo_list = []
    multiprocessing_list = []

    for strategia in ["secuencial", "mpi", "paralelo", "multiprocessing"]:
        args.estrategia = strategia
        args.file1 = "E_coli.fna"
        args.file2 = "Salmonella.fna"
        args.output = "pruebas/pruebas"
        args.outputNoFilter = "pruebas/pruebas_nf"
        if strategia == "secuencial":
                start_time = time.time()
                choose_strategy(args, parser)
                end_time = time.time()
                secuencial_list.append(end_time - start_time)
        else:
            for num_processes in num_processes_list:
                args.num_processes = num_processes
                start_time = time.time()
                choose_strategy(args, parser)
                end_time = time.time()
                if strategia == "mpi":
                    mpi_list.append(end_time - start_time)
                elif strategia == "paralelo":
                    paralelo_list.append(end_time - start_time)
                elif strategia == "multiprocessing":
                    multiprocessing_list.append(end_time - start_time)

if __name__ == "__main__":
    main()
