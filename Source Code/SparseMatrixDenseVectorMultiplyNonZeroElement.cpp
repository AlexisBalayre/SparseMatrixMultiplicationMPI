#include "SparseMatrixDenseVectorMultiplyNonZeroElement.h"
#include <mpi.h>
#include <algorithm>

DenseMatrix sparseMatrixDenseVectorMultiplyNonZeroElement(const SparseMatrix &sparseMatrix, const DenseMatrix &denseVector) {
    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    int numRows = sparseMatrix.rowPtr.size() - 1;
    int numCols = denseVector.size();
    int vecCols = denseVector[0].size();
    int totalNonZeroElements = sparseMatrix.values.size();

    // Calculate the number of non-zero elements each process will handle
    int elementsPerProcess = totalNonZeroElements / worldSize;
    int extraElements = totalNonZeroElements % worldSize;
    int startIdx, endIdx;

    if (worldRank < extraElements) {
        startIdx = worldRank * (elementsPerProcess + 1);
        endIdx = startIdx + elementsPerProcess;
    } else {
        startIdx = worldRank * elementsPerProcess + extraElements;
        endIdx = startIdx + (elementsPerProcess - 1);
    }

    // Compute the multiplication for assigned non-zero elements
    std::vector<double> localResult(numRows * vecCols, 0.0);
    for (int idx = startIdx; idx <= endIdx; ++idx) {
        int row = std::upper_bound(sparseMatrix.rowPtr.begin(), sparseMatrix.rowPtr.end(), idx) - sparseMatrix.rowPtr.begin() - 1;
        int col = sparseMatrix.colIndices[idx];
        double value = sparseMatrix.values[idx];

        for (int k = 0; k < vecCols; ++k) {
            localResult[row * vecCols + k] += value * denseVector[col][k];
        }
    }

    // Flatten the result for MPI_Gather
    std::vector<double> flatResult(numRows * vecCols);
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < vecCols; ++j) {
            flatResult[i * vecCols + j] = localResult[i * vecCols + j];
        }
    }

    // Gather results from all processes at the root
    std::vector<double> gatheredResults;
    if (worldRank == 0) {
        gatheredResults.resize(numRows * vecCols * worldSize);
    }

    MPI_Gather(flatResult.data(), flatResult.size(), MPI_DOUBLE, 
               gatheredResults.data(), flatResult.size(), MPI_DOUBLE, 
               0, MPI_COMM_WORLD);

    // Aggregate results at the root process
    DenseMatrix finalResult(numRows, std::vector<double>(vecCols, 0.0));
    if (worldRank == 0) {
        for (int i = 0; i < worldSize; ++i) {
            for (int j = 0; j < numRows * vecCols; ++j) {
                finalResult[j / vecCols][j % vecCols] += gatheredResults[i * numRows * vecCols + j];
            }
        }
    }

    return (worldRank == 0) ? finalResult : DenseMatrix{};
}
