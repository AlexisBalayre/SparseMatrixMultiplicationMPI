#include <mpi.h>
#include "SparseMatrixDenseVectorMultiplyColumnWise.h"
#include <numeric>  // Pour std::accumulate


/**
 * @brief Function to execute the sparse matrix-dense vector multiplication using column-wise parallel algorithm
 *
 * @param sparseMatrix Sparse matrix
 * @param denseVector Dense vector
 * @param numRow Number of rows in the sparse matrix
 * @param numCols Number of columns in the sparse matrix
 * @param vecCols Number of columns in the dense vector
 * @return DenseVector Result of the multiplication
 */
DenseVector sparseMatrixDenseVectorMultiplyColumnWise(const SparseMatrix &sparseMatrix, const DenseVector &denseVector, int numRows, int numCols, int vecCols)
{
    // MPI Initialisation
    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    // Improved Workload Distribution
    int colsPerProcess = vecCols / worldSize;
    int extraCols = vecCols % worldSize;

    int startCol = worldRank * colsPerProcess;
    int endCol = (worldRank != worldSize - 1) ? startCol + colsPerProcess : startCol + colsPerProcess + extraCols;

    // Efficient Memory Allocation for Local Result
    int localSize = numRows * (endCol - startCol);
    std::vector<double> localResult(localSize, 0.0);

    // Optimized Local Computation
    for (int col = startCol; col < endCol; ++col)
    {
        for (int i = 0; i < numRows; ++i)
        {
            double sum = 0.0;
            for (int j = sparseMatrix.rowPtr[i]; j < sparseMatrix.rowPtr[i + 1]; ++j)
            {
                int sparseCol = sparseMatrix.colIndices[j];
                sum += sparseMatrix.values[j] * denseVector[sparseCol][col];
            }
            localResult[i * (endCol - startCol) + (col - startCol)] = sum;
        }
    }

    // Preparation for Gather operation:
    std::vector<int> recvCounts(worldSize), displacements(worldSize);
    if (worldRank == 0)
    {
        int displacement = 0;
        for (int i = 0; i < worldSize; ++i)
        {
            int startColThisRank = i * colsPerProcess;
            int endColThisRank = (i != worldSize - 1) ? startColThisRank + colsPerProcess : startColThisRank + colsPerProcess + extraCols;
            recvCounts[i] = numRows * (endColThisRank - startColThisRank);
            displacements[i] = displacement;
            displacement += recvCounts[i];
        }
    }

    // Gather all local results into the root process
    std::vector<double> gatheredResults;
    if (worldRank == 0)
    {
        gatheredResults.resize(std::accumulate(recvCounts.begin(), recvCounts.end(), 0));
    }
    MPI_Gatherv(localResult.data(), localSize, MPI_DOUBLE,
                gatheredResults.data(), recvCounts.data(),
                displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Reconstruct the final result matrix in the root process
    DenseVector finalResult;
    if (worldRank == 0)
    {
        finalResult.resize(numRows, std::vector<double>(vecCols, 0.0));
        int resultIndex = 0;
        for (int rank = 0; rank < worldSize; ++rank)
        {
            int numColsThisRank = (rank != worldSize - 1) ? colsPerProcess : colsPerProcess + extraCols;
            int startColThisRank = rank * colsPerProcess;

            for (int row = 0; row < numRows; ++row)
            {
                for (int col = 0; col < numColsThisRank; ++col)
                {
                    finalResult[row][startColThisRank + col] = gatheredResults[resultIndex++];
                }
            }
        }
    }

    return (worldRank == 0)
               ? finalResult
               : DenseVector{};
}