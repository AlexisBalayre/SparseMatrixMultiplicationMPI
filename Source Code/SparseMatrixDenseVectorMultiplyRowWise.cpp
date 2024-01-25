#include <mpi.h>
#include <numeric> // std::accumulate
#include "SparseMatrixDenseVectorMultiplyRowWise.h"

/**
 * @brief Function to multiply a sparse matrix with a dense vector using row-wise distribution
 *
 * @param sparseMatrix  The sparse matrix to be multiplied
 * @param denseVector  The dense vector to be multiplied
 * @param numRows   Number of rows in the sparse matrix
 * @param numCols  Number of columns in the sparse matrix
 * @param vecCols  Number of columns in the dense vector
 * @return DenseVector Result of the multiplication
 */
DenseVector sparseMatrixDenseVectorMultiplyRowWise(const SparseMatrix &sparseMatrix,
                                                   const DenseVector &denseVector,
                                                   int numRows, int numCols, int vecCols)
{
    // MPI Initialisation
    int worldSize, worldRank;                  // Total number of processes and the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize); // Get the total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank); // Get the rank of the current process

    // Calculate the number of non-zero elements in each row for better load distribution
    std::vector<int> nonZeroElementPerRow(numRows); // nonZeroElementPerRow stores the count of non-zero elements in each row
    // Iterate over the rows of the sparse matrix
    for (int i = 0; i < numRows; ++i)
    {
        nonZeroElementPerRow[i] = sparseMatrix.rowPtr[i + 1] - sparseMatrix.rowPtr[i]; // Number of non-zero elements in the current row
    }

    // Distribute rows based on the count of non-zero elements to ensure balanced workload
    std::vector<int> rowsCountPerProcess(worldSize, 0);                                                           // Array to store the count of rows each process will handle
    int totalCountNonZeroElements = std::accumulate(nonZeroElementPerRow.begin(), nonZeroElementPerRow.end(), 0); // Total number of non-zero elements
    int cumCountNonZeroElements = 0;                                                                              // Cumulative count of non-zero elements
    // Iterate over the rows of the sparse matrix
    for (int i = 0; i < numRows; ++i)
    {
        int proc = (cumCountNonZeroElements * worldSize) / totalCountNonZeroElements; // Determine which process will handle this row
        rowsCountPerProcess[proc]++;                                                  // Increment the count of rows for the selected process
        cumCountNonZeroElements += nonZeroElementPerRow[i];                           // Update the cumulative count of non-zero elements
    }

    // Determine the starting and ending rows for the current process
    int startRow = std::accumulate(rowsCountPerProcess.begin(), rowsCountPerProcess.begin() + worldRank, 0); // Cumulative sum of rowsCountPerProcess till the previous process
    int endRow = startRow + rowsCountPerProcess[worldRank] - 1;                                              // Last row assigned to the current process

    // Local computation: each process computes its portion of the result
    DenseVector localResult(rowsCountPerProcess[worldRank], std::vector<double>(vecCols, 0.0));
    // Iterate over the rows assigned to the current process
    for (int i = startRow; i <= endRow; ++i)
    {
        // Iterate over the non-zero elements in the current row
        for (int j = sparseMatrix.rowPtr[i]; j < sparseMatrix.rowPtr[i + 1]; ++j)
        {
            // Iterate over the columns of the dense vector
            for (int k = 0; k < vecCols; ++k)
            {
                localResult[i - startRow][k] += sparseMatrix.values[j] * denseVector[sparseMatrix.colIndices[j]][k]; // Compute the local result
            }
        }
    }

    // Gather operation preparation: collect local result sizes and compute displacements
    std::vector<int> localResultSizes(worldSize), displacements(worldSize);                     // localResultSizes stores the size of the local result in each process
    int localSize = rowsCountPerProcess[worldRank] * vecCols;                                   // Size of the local result
    MPI_Allgather(&localSize, 1, MPI_INT, localResultSizes.data(), 1, MPI_INT, MPI_COMM_WORLD); // Collect local result sizes

    // Flatten the localResult matrix for MPI_Gatherv
    std::vector<double> flatLocalResult(localSize); // flatLocalResult stores the flattened local resul
    // Iterate over the rows of the local result matrix
    for (int i = 0, index = 0; i < rowsCountPerProcess[worldRank]; ++i)
    {
        std::copy(localResult[i].begin(), localResult[i].end(), flatLocalResult.begin() + index); // Copy the current row of localResult into flatLocalResult
        index += vecCols;                                                                         // Update the index
    }

    // Calculate displacements for each process's data in the gathered array
    int displacement = 0;
    for (int i = 0; i < worldSize; ++i)
    {
        displacements[i] = displacement;     // Displacement for the current process
        displacement += localResultSizes[i]; // Update the displacement
    }

    // Gather all local results into the root process
    std::vector<double> gatheredResults;
    // Resize the gatheredResults vector in the root process to accommodate all results
    if (worldRank == 0)
    {
        gatheredResults.resize(displacement); // Resize to accommodate all results
    }
    MPI_Gatherv(flatLocalResult.data(), localSize, MPI_DOUBLE,
                gatheredResults.data(), localResultSizes.data(),
                displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD); // Gather all local results into the root process

    // Reconstruct the final result matrix in the root process
    DenseVector finalResult;
    if (worldRank == 0)
    {
        // Resize the finalResult matrix to accommodate the final result
        finalResult.resize(numRows, std::vector<double>(vecCols, 0.0));
        // Iterate over the rows of the final result matrix
        for (int i = 0, index = 0; i < numRows; ++i)
        {
            // Iterate over the columns of the final result matrix
            for (int j = 0; j < vecCols; ++j, ++index)
            {
                finalResult[i][j] = gatheredResults[index]; // Reconstruct the final result matrix
            }
        }
    }

    // Return the final result in the root process, empty matrix in others
    return (worldRank == 0) ? finalResult : DenseVector{};
}
