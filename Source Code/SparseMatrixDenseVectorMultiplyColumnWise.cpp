#include <mpi.h>
#include "SparseMatrixDenseVectorMultiplyColumnWise.h"

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
    int startCol = worldRank * (colsPerProcess + (worldRank < extraCols ? 1 : 0));
    int endCol = startCol + colsPerProcess + (worldRank < extraCols ? 1 : 0);

    // Efficient Memory Allocation for Local Result
    std::vector<double> localResult(numRows * (endCol - startCol), 0.0);

    // Optimized Local Computation
    for (int col = startCol; col < endCol; ++col)
    {
        for (int i = 0; i < numRows; ++i)
        {
            double sum = 0.0;
            for (int j = sparseMatrix.rowPtr[i]; j < sparseMatrix.rowPtr[i + 1]; ++j)
            {
                int sparseCol = sparseMatrix.colIndices[j];
                if (sparseCol < numCols)
                {
                    sum += sparseMatrix.values[j] * denseVector[sparseCol][col];
                }
            }
            localResult[i * (endCol - startCol) + (col - startCol)] = sum;
        }
    }

    // Gather operation preparation:
    std::vector<int> localResultSizes(worldSize), displacements(worldSize);
    int localSize = numRows * (endCol - startCol);
    MPI_Allgather(&localSize, 1, MPI_INT, localResultSizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
 
    // Calculate displacements for each process's data in the gathered array
    int displacement = 0;
    for (int i = 0; i < worldSize; ++i)
    {
        displacements[i] = displacement;
        displacement += localResultSizes[i];
    }

    // Gather all local results into the root process
    std::vector<double> gatheredResults;
    if (worldRank == 0)
    {
        gatheredResults.resize(displacement);
    }
    MPI_Gatherv(localResult.data(), localSize, MPI_DOUBLE,
                gatheredResults.data(), localResultSizes.data(),
                displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Remodeler le résultat global en DenseVector
    // Reconstruct the final result matrix in the root process
    DenseVector finalResult(numRows, std::vector<double>(vecCols, 0.0));
    if (worldRank == 0)
    {
        int resultIndex = 0;
        for (int rank = 0; rank < worldSize; ++rank)
        {
            int numColsThisRank = (rank < extraCols) ? (colsPerProcess + 1) : colsPerProcess;
            int startColThisRank = rank * colsPerProcess + std::min(rank, extraCols);

            for (int row = 0; row < numRows; ++row)
            {
                for (int col = 0; col < numColsThisRank; ++col)
                {
                    finalResult[row][startColThisRank + col] = gatheredResults[resultIndex++];
                }
            }
        }
    }

    return (worldRank == 0) ? finalResult : DenseVector{};
}