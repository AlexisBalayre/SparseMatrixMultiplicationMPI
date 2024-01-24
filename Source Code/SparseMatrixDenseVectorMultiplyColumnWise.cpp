#include <mpi.h>
#include "SparseMatrixDenseVectorMultiplyColumnWise.h"


// Assuming the definitions of SparseMatrix and DenseMatrix are provided elsewhere
// DenseMatrix is assumed to be a 2D vector: std::vector<std::vector<double>>

DenseMatrix sparseMatrixDenseVectorMultiplyColumnWise(const SparseMatrix &sparseMatrix, const DenseMatrix &denseVector, int numRows, int numCols, int vecCols) {
    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    int colsPerProcess = vecCols / worldSize;
    int extraCols = vecCols % worldSize;
    int startCol, endCol;

    if (worldRank < extraCols) {
        startCol = worldRank * (colsPerProcess + 1);
        endCol = startCol + colsPerProcess + 1;
    } else {
        startCol = worldRank * colsPerProcess + extraCols;
        endCol = startCol + colsPerProcess;
    }

    DenseMatrix localResult(numRows, std::vector<double>(endCol - startCol, 0.0));

    // Optimized column-wise multiplication
    for (int col = startCol; col < endCol; ++col) {
        for (int i = 0; i < numRows; ++i) {
            double sum = 0.0;
            for (int j = sparseMatrix.rowPtr[i]; j < sparseMatrix.rowPtr[i + 1]; ++j) {
                int sparseCol = sparseMatrix.colIndices[j];
                if (sparseCol < numCols) {
                    sum += sparseMatrix.values[j] * denseVector[sparseCol][col];
                }
            }
            localResult[i][col - startCol] = sum;
        }
    }

    // Optimized memory allocation for flatLocalResult
    std::vector<double> flatLocalResult;
    flatLocalResult.reserve(numRows * (endCol - startCol));
    for (const auto &row : localResult) {
        flatLocalResult.insert(flatLocalResult.end(), row.begin(), row.end());
    }

    int sendCount = flatLocalResult.size();

    std::vector<double> gatheredResults;
    std::vector<int> receiveCounts(worldSize), displacements(worldSize);

    if (worldRank == 0) {
        gatheredResults.resize(numRows * vecCols);

        int displacement = 0;
        for (int i = 0; i < worldSize; ++i) {
            int numColsProcessed = (i < extraCols) ? (colsPerProcess + 1) : colsPerProcess;
            receiveCounts[i] = numRows * numColsProcessed;
            displacements[i] = displacement;
            displacement += receiveCounts[i];
        }
    }

    MPI_Gatherv(flatLocalResult.data(), sendCount, MPI_DOUBLE,
                gatheredResults.data(), receiveCounts.data(), displacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    DenseMatrix finalResult;
    if (worldRank == 0) {
        finalResult.resize(numRows, std::vector<double>(vecCols, 0.0));
        int resultIndex = 0;

        for (int rank = 0; rank < worldSize; ++rank) {
            int numColsThisRank = (rank < extraCols) ? (colsPerProcess + 1) : colsPerProcess;
            int startColThisRank = rank * colsPerProcess + std::min(rank, extraCols);

            for (int row = 0; row < numRows; ++row) {
                for (int col = 0; col < numColsThisRank; ++col) {
                    finalResult[row][startColThisRank + col] = gatheredResults[resultIndex++];
                }
            }
        }
    }

    return (worldRank == 0) ? finalResult : DenseMatrix{};
}
