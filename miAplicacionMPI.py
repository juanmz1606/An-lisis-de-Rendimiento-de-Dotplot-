import argparse

def main(file1, file2, threshold, output):
    # Aquí iría la lógica de tu aplicación
    print(f"Procesando {file1} y {file2} con un umbral de {threshold}. Guardando resultado en {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aplicación MPI")
    parser.add_argument("--file1", required=True, help="Primer archivo de entrada")
    parser.add_argument("--file2", required=True, help="Segundo archivo de entrada")
    parser.add_argument("--thres", type=float, required=True, help="Umbral")
    parser.add_argument("--output", required=True, help="Archivo de salida")

    args = parser.parse_args()

    main(args.file1, args.file2, args.thres, args.output)
