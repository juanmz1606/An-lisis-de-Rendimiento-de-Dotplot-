import subprocess
import argparse

def run_mpi(num_processes, file1, file2, output,output_nf):
    command = [
        "mpirun",
        "-n", str(num_processes),
        "python3", "src/miAplicacionMPI.py",
        f"--file1={file1}",
        f"--file2={file2}",
        f"--output={output}",
        f"--outputNoFilter={output_nf}"
    ]
    subprocess.run(command)
    
def run_paralelo(num_processes, file1, file2, output,output_nf):
    command = [
        "mpirun",
        "-n", str(num_processes),
        "python3", "src/miAplicacionMPI.py",
        f"--file1={file1}",
        f"--file2={file2}",
        f"--output={output}",
        f"--outputNoFilter={output_nf}"
    ]
    subprocess.run(command)

def run_secuencial(file1, file2, output,output_nf):
    command = [
        "python3", "src/secuencial.py",
        f"--file1={file1}",
        f"--file2={file2}",
        f"--output={output}",
        f"--outputNoFilter={output_nf}"
    ]
    subprocess.run(command)

def main():
    parser = argparse.ArgumentParser(description="Script para ejecutar la aplicación")
    parser.add_argument("-e", "--estrategia", choices=["mpi", "secuencial", "paralelo", "multiprocessing", "pycuda"], required=True, help="Estrategia de ejecución")
    parser.add_argument("-n", "--num_processes", type=int, required=False, help="Número de procesos")
    parser.add_argument("-f1", "--file1", required=True, help="Primer archivo de entrada")
    parser.add_argument("-f2", "--file2", required=True, help="Segundo archivo de entrada")
    parser.add_argument("-t", "--thres", type=float, required=True, help="Umbral")
    parser.add_argument("-o", "--output", required=True, help="Archivo de salida")
    parser.add_argument("-outnf", "--outputNoFilter", type=str, required=True, help="Archivo de salida sin filtro")

    args = parser.parse_args()

    if args.estrategia == "mpi":
        if args.num_processes is None:
            parser.error("--num_processes es requerido para la estrategia MPI")
        run_mpi(args.num_processes, args.file1, args.file2, args.output, args.outputNoFilter)
    elif args.estrategia == "secuencial":
        run_secuencial(args.file1, args.file2, args.output, args.outputNoFilter)
    elif args.estrategia == "paralelo":
        run_paralelo(args.num_processes,args.file1, args.file2, args.output, args.outputNoFilter)

if __name__ == "__main__":
    main()
