import subprocess
import argparse

def run_mpi(num_processes, file1, file2, output, output_nf, num_seqs=None):
    command = [
        "mpirun",
        "-n", str(num_processes),
        "python3", "src/mpi_strategy.py",
        f"--file1={file1}",
        f"--file2={file2}",
        f"--num_seqs={num_seqs}" if num_seqs else ""
        f"--output={output}",
        f"--outputNoFilter={output_nf}"
    ]
    subprocess.run(command)

def run_secuencial(file1, file2, output, output_nf, num_seqs=1000):
    command = [
        "python3", "src/secuencial.py",
        f"--file1={file1}",
        f"--file2={file2}",
        f"--num_seqs={num_seqs}" if num_seqs else "",
        f"--output={output}",
        f"--outputNoFilter={output_nf}"
    ]
    subprocess.run(command)

def main():
    parser = argparse.ArgumentParser(description="Script para ejecutar la aplicación")
    parser.add_argument("-e", "--estrategia", choices=["mpi", "hilos", "secuencial", "multiprocessing", "pycuda"], required=True, help="Estrategia de ejecución")
    parser.add_argument("-n", "--num_processes", type=int, required=False, help="Número de procesos para mpiexec (requerido solo para estrategia MPI)")
    parser.add_argument("-f1", "--file1", required=True, help="Primer archivo de entrada")
    parser.add_argument("-f2", "--file2", required=True, help="Segundo archivo de entrada")
    parser.add_argument("-t", "--thres", type=float, required=True, help="Umbral")
    parser.add_argument("-o", "--output", required=True, help="Archivo de salida")
    parser.add_argument("-outnf", "--outputNoFilter", type=str, required=True, help="Archivo de salida sin filtro")
    parser.add_argument("-c", "--num_seqs", type=int, default=100, help="Número de caracteres a tomar de cada secuencia")

    args = parser.parse_args()

    if args.estrategia == "mpi":
        if args.num_processes is None:
            parser.error("--num_processes es requerido para la estrategia MPI")
        run_mpi(args.num_processes, args.file1, args.file2, args.output, args.num_seqs)
    else:
        run_secuencial(args.file1, args.file2, args.output, args.outputNoFilter)

if __name__ == "__main__":
    main()
