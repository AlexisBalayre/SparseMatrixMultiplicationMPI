#include "SparseMatrixDenseVectorMultiplyNonZeroElement.h"
#include <mpi.h>

/**
 * @brief Function to execute the sparse matrix-dense vector multiplication using non-zero element parallel algorithm
 *
 * @param sparseMatrix  Sparse matrix
 * @param denseVector  Dense vector
 * @param numRows  Number of rows in the sparse matrix
 * @param numCols  Number of columns in the sparse matrix
 * @param vecCols  Number of columns in the dense vector
 * @return DenseVector  Result of the multiplication
 */
DenseVector sparseMatrixDenseVectorMultiplyNonZeroElement(const SparseMatrix &sparseMatrix, const DenseVector &denseVector, int numRows, int numCols, int vecCols)
{
    // MPI Initialisation
    int worldSize, worldRank;                  // Number of processes and rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize); // Get the total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank); // Get the rank of the current process

    // Distribute non-zero elements among processes
    int totalNonZeroElements = sparseMatrix.values.size();     // Total number of non-zero elements
    int elementsPerProcess = totalNonZeroElements / worldSize; // Number of non-zero elements to be processed by each process
    int extraElements = totalNonZeroElements % worldSize;      // Number of processes that will process one extra non-zero element
    int startIdx, endIdx;                                      // Starting and ending indices of the non-zero elements for the current process

    // Determine the starting and ending indices of the non-zero elements for the current process
    if (worldRank < extraElements)
    {
        startIdx = worldRank * (elementsPerProcess + 1);
        endIdx = startIdx + elementsPerProcess + 1;
    }
    else
    {
        startIdx = worldRank * elementsPerProcess + extraElements;
        endIdx = startIdx + elementsPerProcess;
    }

    // Precompute the row index mapping
    std::vector<int> rowIndexMap(sparseMatrix.values.size()); // Stores the row index of each non-zero element
    // Iterate over the rows of the sparse matrix
    for (int row = 0, idx = 0; row < sparseMatrix.rowPtr.size() - 1; ++row)
    {
        // Iterate over the non-zero elements in the current row
        for (; idx < sparseMatrix.rowPtr[row + 1]; ++idx)
        {
            rowIndexMap[idx] = row; // Store the row index of the current non-zero element
        }
    }

    // Compute the multiplication for assigned non-zero elements
    std::vector<double> localResult(numRows * vecCols, 0.0); // Local result vector
    // Iterate over the non-zero elements assigned to the current process
    for (int idx = startIdx; idx < endIdx; ++idx)
    {
        int row = rowIndexMap[idx];              // Row index of the current non-zero element
        int col = sparseMatrix.colIndices[idx];  // Column index of the current non-zero element
        double value = sparseMatrix.values[idx]; // Value of the current non-zero element

        // Iterate over the columns of the dense vector
        for (int k = 0; k < vecCols; ++k)
        {
            localResult[row * vecCols + k] += value * denseVector[col][k]; // Compute the local result
        }
    }

    // Initialize the final result only in the root process
    DenseVector finalResult;
    if (worldRank == 0)
    {
        finalResult.resize(numRows, std::vector<double>(vecCols, 0.0)); // Initialize the final result vector
    }

    // Gather the local results in the root process
    std::vector<double> flatFinalResult(numRows * vecCols, 0.0);                                                       // Flat vector to store the final result
    MPI_Reduce(localResult.data(), flatFinalResult.data(), numRows * vecCols, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // Gather the local results in the root process

    // Reconstruct the finalResult from flatFinalResult in the root process
    if (worldRank == 0)
    {
        // Iterate over the rows of the final result
        for (int i = 0; i < numRows; ++i)
        {
            std::copy(flatFinalResult.begin() + i * vecCols, flatFinalResult.begin() + (i + 1) * vecCols, finalResult[i].begin()); // Copy the current row of flatFinalResult into the final result
        }
    }

    return (worldRank == 0) ? finalResult : DenseVector{}; // Return the final result only in the root process
}