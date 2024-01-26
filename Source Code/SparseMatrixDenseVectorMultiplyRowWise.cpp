#include <mpi.h>
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

    int rowsCountPerProcess = numRows / worldSize; // Number of rows to be processed by each process
    int extraRows = numRows % worldSize;           // Number of processes that will process one extra row

    int startRow = worldRank * rowsCountPerProcess + std::min(worldRank, extraRows);
    int endRow = startRow + rowsCountPerProcess + (worldRank < extraRows ? 1 : 0);

    // Local computation: each process computes its portion of the result
    int localSize = (endRow - startRow) * vecCols;
    std::vector<double> localResult(localSize);

    for (int i = startRow; i < endRow; ++i)
    {
        for (int j = sparseMatrix.rowPtr[i]; j < sparseMatrix.rowPtr[i + 1]; ++j)
        {
            int colIndex = sparseMatrix.colIndices[j];
            for (int k = 0; k < vecCols; ++k)
            {
                int localIndex = (i - startRow) * vecCols + k;
                localResult[localIndex] += sparseMatrix.values[j] * denseVector[colIndex][k];
            }
        }
    }

    // Préparation pour MPI_Gatherv
    std::vector<int> recvCounts(worldSize), displacements(worldSize);
    if (worldRank == 0)
    {
        int totalSize = 0;
        for (int rank = 0; rank < worldSize; ++rank)
        {
            int startRowThisRank = rank * rowsCountPerProcess + std::min(rank, extraRows);
            int endRowThisRank = startRowThisRank + rowsCountPerProcess + (rank < extraRows ? 1 : 0);
            recvCounts[rank] = (endRowThisRank - startRowThisRank) * vecCols;
            displacements[rank] = totalSize;
            totalSize += recvCounts[rank];
        }
    }

    // Gather all local results into the root process
    std::vector<double> gatheredResults;
    if (worldRank == 0)
    {
        gatheredResults.resize(recvCounts[0] * worldSize); // Assurez-vous que la taille est suffisante
    }
    MPI_Gatherv(localResult.data(), localSize, MPI_DOUBLE,
                gatheredResults.data(), recvCounts.data(),
                displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Reconstruct the final result matrix in the root process
    DenseVector finalResult;
    if (worldRank == 0)
    {
        finalResult.resize(numRows, std::vector<double>(vecCols, 0.0)); // Initialize the final result vector
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
