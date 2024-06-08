import argparse

def run_secuencial(file1, file2, threshold, output):
    print("Ejecutando la aplicación de forma secuencial")
    print(f"file1: {file1}")
    print(f"file2: {file2}")
    print(f"threshold: {threshold}")
    print(f"output: {output}")

    # Aquí iría el código de tu aplicación secuencial
    # ...

def main():
    parser = argparse.ArgumentParser(description="Aplicación secuencial")
    parser.add_argument("--file1", required=True, help="Primer archivo de entrada")
    parser.add_argument("--file2", required=True, help="Segundo archivo de entrada")
    parser.add_argument("--thres", type=float, required=True, help="Umbral")
    parser.add_argument("--output", required=True, help="Archivo de salida")

    args = parser.parse_args()

    run_secuencial(args.file1, args.file2, args.thres, args.output)

if __name__ == "__main__":
    main()