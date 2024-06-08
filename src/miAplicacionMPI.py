import argparse

def run_application(file1, file2, threshold, output):
    # Aquí iría el código de tu aplicación MPI
    print(f"Ejecutando la aplicación con los siguientes argumentos:")
    print(f"file1: {file1}")
    print(f"file2: {file2}")
    print(f"threshold: {threshold}")
    print(f"output: {output}")

def main():
    parser = argparse.ArgumentParser(description="Aplicación MPI")
    parser.add_argument("--file1", required=True, help="Primer archivo de entrada")
    parser.add_argument("--file2", required=True, help="Segundo archivo de entrada")
    parser.add_argument("--thres", type=float, required=True, help="Umbral")
    parser.add_argument("--output", required=True, help="Archivo de salida")

    args = parser.parse_args()

    run_application(args.file1, args.file2, args.thres, args.output)

if __name__ == "__main__":
    main()