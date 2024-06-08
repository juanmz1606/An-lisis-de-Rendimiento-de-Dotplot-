import subprocess
import argparse

def run_mpi(num_processes, file1, file2, threshold, output):
    command = [
        "mpirun",
        "-n", str(num_processes),
        "python3", "src/miAplicacionMPI.py",
        f"--file1={file1}",
        f"--file2={file2}",
        f"--thres={threshold}",
        f"--output={output}"
    ]
    #print("Comando a ejecutar:", " ".join(command))
    subprocess.run(command)

def run_secuencial(file1, file2, threshold, output):
    command = [
        "python3", "src/secuencial.py",
        f"--file1={file1}",
        f"--file2={file2}",
        f"--thres={threshold}",
        f"--output={output}"
    ]
    #print("Comando a ejecutar:", " ".join(command))
    subprocess.run(command)

def main():
    parser = argparse.ArgumentParser(description="Script para ejecutar la aplicación")
    parser.add_argument("-e", "--estrategia", choices=["mpi", "secuencial", "multiprocessing", "pycuda"], required=True, help="Estrategia de ejecución")
    parser.add_argument("-n", "--num_processes", type=int, required=False, help="Número de procesos para mpiexec (requerido solo para estrategia MPI)")
    parser.add_argument("-f1", "--file1", required=True, help="Primer archivo de entrada")
    parser.add_argument("-f2", "--file2", required=True, help="Segundo archivo de entrada")
    parser.add_argument("-t", "--thres", type=float, required=True, help="Umbral")
    parser.add_argument("-o", "--output", required=True, help="Archivo de salida")

    args = parser.parse_args()

    if args.estrategia == "mpi":
        if args.num_processes is None:
            parser.error("--num_processes es requerido para la estrategia MPI")
        run_mpi(args.num_processes, args.file1, args.file2, args.thres, args.output)
    else:
        run_secuencial(args.file1, args.file2, args.thres, args.output)

if __name__ == "__main__":
    main()
