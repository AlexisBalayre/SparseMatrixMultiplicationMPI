#include <iostream>
#include <vector>
#include <mpi.h>

std::vector<double> sparseMatrixVectorMultiplyParallel(const std::vector<double> &values,
                                                       const std::vector<int> &rows,
                                                       const std::vector<int> &cols,
                                                       const std::vector<double> &vecteur,
                                                       int numRows, int startRow, int endRow) {
    std::vector<double> result(numRows, 0.0);

    for (int i = startRow; i < endRow; ++i) {
        for (int j = rows[i]; j < rows[i + 1]; ++j) {
            result[i - startRow] += values[j] * vecteur[cols[j]];
        }
    }

    return result;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Assuming the matrix is distributed or can be distributed
    // Example data (this should be distributed appropriately)
    std::vector<double> values = {1, 2, 3, 4}; 
    std::vector<int> rows = {0, 2, 3, 3, 4};   
    std::vector<int> cols = {0, 2, 2, 3};      
    std::vector<double> vecteur = {1, 2, 3, 4};

    // Determine the rows each process will work on
    int numRows = 4; // Total number of rows
    int rowsPerProc = numRows / size;
    int startRow = rank * rowsPerProc;
    int endRow = (rank == size - 1) ? numRows : startRow + rowsPerProc;

    // Call the parallel multiplication function
    std::vector<double> localResult = sparseMatrixVectorMultiplyParallel(values, rows, cols, vecteur, rowsPerProc, startRow, endRow);

    // Gather the results from all processes
    std::vector<double> globalResult;
    if (rank == 0) {
        globalResult.resize(numRows);
    }
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
