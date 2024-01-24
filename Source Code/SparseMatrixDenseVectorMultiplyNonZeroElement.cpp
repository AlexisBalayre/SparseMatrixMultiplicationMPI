#include "SparseMatrixDenseVectorMultiplyNonZeroElement.h"
#include <mpi.h>

DenseMatrix sparseMatrixDenseVectorMultiplyNonZeroElement(const SparseMatrix &sparseMatrix, const DenseMatrix &denseVector, int numRows, int numCols, int vecCols)
{
    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    // Distribute non-zero elements among processes
    int totalNonZeroElements = sparseMatrix.values.size();
    int elementsPerProcess = totalNonZeroElements / worldSize;
    int extraElements = totalNonZeroElements % worldSize;
    int startIdx, endIdx;

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
    std::vector<int> rowIndexMap(sparseMatrix.values.size());
    for (int row = 0, idx = 0; row < sparseMatrix.rowPtr.size() - 1; ++row)
    {
        for (; idx < sparseMatrix.rowPtr[row + 1]; ++idx)
        {
            rowIndexMap[idx] = row;
        }
    }

    // Compute the multiplication for assigned non-zero elements
    std::vector<double> localResult(numRows * vecCols, 0.0);
    for (int idx = startIdx; idx < endIdx; ++idx)
    {
        int row = rowIndexMap[idx];
        int col = sparseMatrix.colIndices[idx];
        double value = sparseMatrix.values[idx];

        for (int k = 0; k < vecCols; ++k)
        {
            localResult[row * vecCols + k] += value * denseVector[col][k];
        }
    }

    // Initialize the final result only in the root process
    DenseMatrix finalResult;
    if (worldRank == 0)
    {
        finalResult.resize(numRows, std::vector<double>(vecCols, 0.0));
    }

    std::vector<double> flatFinalResult(numRows * vecCols, 0.0);
    MPI_Reduce(localResult.data(), flatFinalResult.data(), numRows * vecCols, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Reconstruct the finalResult from flatFinalResult in the root process
    if (worldRank == 0)
    {
        for (int i = 0; i < numRows; ++i)
        {
            std::copy(flatFinalResult.begin() + i * vecCols, flatFinalResult.begin() + (i + 1) * vecCols, finalResult[i].begin());
        }
    }

    return (worldRank == 0) ? finalResult : DenseMatrix{};
}