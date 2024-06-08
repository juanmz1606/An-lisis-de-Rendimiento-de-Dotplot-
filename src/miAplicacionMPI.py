import argparse
from mpi4py import MPI
import numpy as np
from Bio import SeqIO
import matplotlib.pyplot as plt
import time

def load_fasta(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
    return sequences

def generate_dotplot(seq1, seq2):
    n = len(seq1)
    m = len(seq2)
    dotplot = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            if seq1[i] == seq2[j]:
                dotplot[i, j] = 1
    return dotplot

def save_dotplot(dotplot, output_file):
    plt.imshow(dotplot, cmap='Greys', interpolation='nearest')
    plt.xlabel("Sequence 1")
    plt.ylabel("Sequence 2")
    plt.savefig(output_file)

def mpi_dotplot(file1, file2, threshold, output):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        seq1 = load_fasta(file1)[0]
        seq2 = load_fasta(file2)[0]
        n = len(seq1)
        m = len(seq2)
    else:
        seq1 = None
        seq2 = None
        n = None
        m = None

    seq1 = comm.bcast(seq1, root=0)
    seq2 = comm.bcast(seq2, root=0)
    n = comm.bcast(n, root=0)
    m = comm.bcast(m, root=0)

    local_n = n // size
    local_seq1 = seq1[rank*local_n:(rank+1)*local_n]

    local_dotplot = generate_dotplot(local_seq1, seq2)

    if rank == 0:
        dotplot = np.zeros((n, m))
    else:
        dotplot = None

    comm.Gather(local_dotplot, dotplot, root=0)

    if rank == 0:
        save_dotplot(dotplot, output)

def main():
    parser = argparse.ArgumentParser(description="Aplicación MPI para Dotplot")
    parser.add_argument("--file1", required=True, help="Primer archivo de entrada")
    parser.add_argument("--file2", required=True, help="Segundo archivo de entrada")
    parser.add_argument("--thres", type=float, required=True, help="Umbral")
    parser.add_argument("--output", required=True, help="Archivo de salida")

    args = parser.parse_args()

    mpi_dotplot(args.file1, args.file2, args.thres, args.output)

if __name__ == "__main__":
    main()
