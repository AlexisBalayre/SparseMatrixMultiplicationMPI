#include <mpi.h>
#include <numeric> // std::accumulate
#include "SparseMatrixDenseVectorMultiplyRowWise.h"

DenseMatrix sparseMatrixDenseVectorMultiplyRowWise(const SparseMatrix &sparseMatrix,
                                                   const DenseMatrix &denseVector,
                                                   int numRows, int numCols, int vecCols)
{

    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    // Improved Load Balancing
    int rowsPerProcess = numRows / worldSize;
    int extraRows = numRows % worldSize;
    int startRow, endRow;

    if (worldRank < extraRows)
    {
        startRow = worldRank * (rowsPerProcess + 1);
        endRow = startRow + rowsPerProcess;
    }
    else
    {
        startRow = worldRank * rowsPerProcess + extraRows;
        endRow = startRow + (rowsPerProcess - 1);
    }

    // Local computation for each process
    DenseMatrix localResult(endRow - startRow + 1, std::vector<double>(vecCols, 0.0));
    for (int i = startRow; i <= endRow; ++i)
    {
        for (int j = sparseMatrix.rowPtr[i]; j < sparseMatrix.rowPtr[i + 1]; ++j)
        {
            for (int k = 0; k < vecCols; ++k)
            {
                localResult[i - startRow][k] += sparseMatrix.values[j] * denseVector[sparseMatrix.colIndices[j]][k];
            }
        }
    }

    // Gather results at the main process
    std::vector<int> localResultSizes(worldSize), displacements(worldSize);
    int localSize = (endRow - startRow + 1) * vecCols;
    MPI_Gather(&localSize, 1, MPI_INT, localResultSizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Prepare for gathering the results
    std::vector<double> gatheredResults;
    if (worldRank == 0)
    {
        int totalSize = std::accumulate(localResultSizes.begin(), localResultSizes.end(), 0);
        gatheredResults.resize(totalSize);
    }

    // Flatten the localResult matrix for MPI_Gatherv
    std::vector<double> flatLocalResult(localResult.size() * vecCols);
    for (int i = 0; i < localResult.size(); ++i)
    {
        std::copy(localResult[i].begin(), localResult[i].end(), flatLocalResult.begin() + i * vecCols);
    }

    // Calculate displacements for MPI_Gatherv
    if (worldRank == 0)
    {
        int displacement = 0;
        for (int i = 0; i < worldSize; ++i)
        {
            displacements[i] = displacement;
            displacement += localResultSizes[i];
        }
    }

    // Gather the flattened local results
    MPI_Gatherv(flatLocalResult.data(), localSize, MPI_DOUBLE, gatheredResults.data(), localResultSizes.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Reconstruct the final result matrix at the main process
    DenseMatrix finalResult;
    if (worldRank == 0)
    {
        finalResult.resize(numRows, std::vector<double>(vecCols, 0.0));
        for (int i = 0, index = 0; i < numRows; ++i)
        {
            for (int j = 0; j < vecCols; ++j, ++index)
            {
                finalResult[i][j] = gatheredResults[index];
            }
        }
    }

    return (worldRank == 0) ? finalResult : DenseMatrix{};
}
