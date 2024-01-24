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
    // Each process works on approximately equal number of non-zero elements
    std::vector<int> nnzPerRow(numRows);
    for (int i = 0; i < numRows; ++i)
    {
        nnzPerRow[i] = sparseMatrix.rowPtr[i + 1] - sparseMatrix.rowPtr[i];
    }

    // Distribute the rows based on non-zero elements count
    std::vector<int> rowsToProcess(worldSize, 0);
    int totalNnz = std::accumulate(nnzPerRow.begin(), nnzPerRow.end(), 0);
    int cumNnz = 0;
    for (int i = 0; i < numRows; ++i)
    {
        int proc = (cumNnz * worldSize) / totalNnz;
        rowsToProcess[proc]++;
        cumNnz += nnzPerRow[i];
    }

    int startRow = std::accumulate(rowsToProcess.begin(), rowsToProcess.begin() + worldRank, 0);
    int endRow = startRow + rowsToProcess[worldRank] - 1;

    // Local computation for each process
    DenseMatrix localResult(rowsToProcess[worldRank], std::vector<double>(vecCols, 0.0));
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

    // Optimize data communication using MPI_Gatherv
    // Gather the sizes of the local results first
    std::vector<int> localResultSizes(worldSize), displacements(worldSize);
    int localSize = rowsToProcess[worldRank] * vecCols;
    MPI_Allgather(&localSize, 1, MPI_INT, localResultSizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Flatten the localResult matrix for MPI_Gatherv
    std::vector<double> flatLocalResult(localSize);
    for (int i = 0, index = 0; i < rowsToProcess[worldRank]; ++i)
    {
        std::copy(localResult[i].begin(), localResult[i].end(), flatLocalResult.begin() + index);
        index += vecCols;
    }

    // Calculate displacements for MPI_Gatherv
    int displacement = 0;
    for (int i = 0; i < worldSize; ++i)
    {
        displacements[i] = displacement;
        displacement += localResultSizes[i];
    }

    // Gather the flattened local results in the root process
    std::vector<double> gatheredResults;
    if (worldRank == 0)
    {
        gatheredResults.resize(displacement);
    }
    MPI_Gatherv(flatLocalResult.data(), localSize, MPI_DOUBLE,
                gatheredResults.data(), localResultSizes.data(),
                displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

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
