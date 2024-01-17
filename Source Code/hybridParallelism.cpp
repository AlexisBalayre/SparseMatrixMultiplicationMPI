#include <iostream>
#include <vector>
#include <mpi.h>

struct NonZeroElement {
    double value;
    int row;
    int col;
};

std::vector<double> MPIHybridParallelMult(const std::vector<NonZeroElement>& chunkNonZeroElements,
                                          const std::vector<double>& vecteur,
                                          int startRow, int endRow) {
    std::vector<double> r_local(endRow - startRow, 0.0);

    for (const auto& elem : chunkNonZeroElements) {
        if (elem.row >= startRow && elem.row < endRow) {
            r_local[elem.row - startRow] += elem.value * vecteur[elem.col];
        }
    }

    return r_local;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Example data, distributed among processes
    std::vector<NonZeroElement> nonZeroElements; // Populate and distribute among processes
    std::vector<double> vecteur = {1, 2, 3, 4}; // Broadcast to all processes

    int numRows = 4; // Total number of rows
    int rowsPerProc = numRows / size;
    int startRow = rank * rowsPerProc;
    int endRow = (rank == size - 1) ? numRows : startRow + rowsPerProc;

    std::vector<NonZeroElement> chunkNonZeroElements = /* Logic to distribute non-zero elements within row chunks */;

    std::vector<double> localResult = MPIHybridParallelMult(chunkNonZeroElements, vecteur, startRow, endRow);

    // Gather all local results into the final result vector
    std::vector<double> globalResult(numRows, 0.0);
    MPI_Gather(localResult.data(), rowsPerProc, MPI_DOUBLE,
               globalResult.data(), rowsPerProc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Display the result
        for (double val : globalResult) {
            std::cout << val << " ";
        }
    }

    MPI_Finalize();
    return 0;
}
