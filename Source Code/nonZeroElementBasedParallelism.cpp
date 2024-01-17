#include <iostream>
#include <vector>
#include <mpi.h>

// Structure to represent non-zero elements
struct NonZeroElement {
    double value;
    int row;
    int col;
};

std::vector<double> MPIElementBasedMult(const std::vector<NonZeroElement>& nonZeroElementsSubset,
                                        const std::vector<double>& vecteur,
                                        int numRows) {
    std::vector<double> result(numRows, 0.0);

    for (const auto& elem : nonZeroElementsSubset) {
        result[elem.row] += elem.value * vecteur[elem.col];
    }

    return result;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Example data (Distributed appropriately among processes)
    std::vector<NonZeroElement> nonZeroElements; // Populate with non-zero elements
    std::vector<double> vecteur = {1, 2, 3, 4}; // Broadcast to all processes

    // Divide non-zero elements among processes
    // Each process works on its subset of non-zero elements
    std::vector<NonZeroElement> nonZeroElementsSubset = /* Logic to distribute non-zero elements */;

    std::vector<double> localResult = MPIElementBasedMult(nonZeroElementsSubset, vecteur, 4);

    // Reduce all local results to form the final result vector
    std::vector<double> globalResult(4, 0.0);
    MPI_Reduce(localResult.data(), globalResult.data(), 4, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Display the result
        for (double val : globalResult) {
            std::cout << val << " ";
        }
    }

    MPI_Finalize();
    return 0;
}
