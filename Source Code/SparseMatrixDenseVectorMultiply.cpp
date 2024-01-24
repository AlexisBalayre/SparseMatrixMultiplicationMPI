#include "SparseMatrixDenseVectorMultiply.h"

// Implementation of the sparseMatrixDenseVectorMultiply function
DenseMatrix sparseMatrixDenseVectorMultiply(const SparseMatrix &sparseMatrix,
                                            const DenseMatrix &denseVector,
                                            int numRows, int numCols, int vecCols) {
    DenseMatrix result(numRows, std::vector<double>(vecCols, 0.0));
    for (int i = 0; i < numRows; ++i) {
        for (int j = sparseMatrix.rowPtr[i]; j < sparseMatrix.rowPtr[i + 1]; ++j) {
            for (int k = 0; k < vecCols; ++k) {
                result[i][k] += sparseMatrix.values[j] * denseVector[sparseMatrix.colIndices[j]][k];
            }
        }
    }
    return result;
}