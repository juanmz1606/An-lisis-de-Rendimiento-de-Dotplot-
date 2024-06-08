import subprocess
import argparse
import os
import shutil
import sys

def run_application(num_processes, file1, file2, threshold, output):
    # Construir el comando mpiexec
    command = [
        "mpiexec",
        "-n", str(num_processes),
        "python", os.path.abspath("miAplicacionMPI.py"),  # Asegurarse de usar la ruta absoluta
        f"--file1={file1}",
        f"--file2={file2}",
        f"--thres={threshold}",
        f"--output={output}"
    ]

    # Imprimir el comando para depuración
    print("Comando a ejecutar:", " ".join(command))

    # Ejecutar el comando
    result = subprocess.run(command, capture_output=True, text=True)

    # Imprimir la salida
    if result.returncode == 0:
        print("Ejecución exitosa")
        print(result.stdout)
    else:
        print("Error en la ejecución")
        print(result.stderr)

def main():
    parser = argparse.ArgumentParser(description="Script para ejecutar miAplicacionMPI.py con mpiexec")
    parser.add_argument("--num_processes", type=int, required=True, help="Número de procesos para mpiexec")
    parser.add_argument("--file1", required=True, help="Primer archivo de entrada")
    parser.add_argument("--file2", required=True, help="Segundo archivo de entrada")
    parser.add_argument("--thres", type=float, required=True, help="Umbral")
    parser.add_argument("--output", required=True, help="Archivo de salida")

    args = parser.parse_args()

    # Verificar que mpiexec esté en el PATH
    if not shutil.which("mpiexec"):
        print("Error: mpiexec no está en el PATH del sistema.")
        sys.exit(1)

    run_application(args.num_processes, args.file1, args.file2, args.thres, args.output)

if __name__ == "__main__":
    main()
