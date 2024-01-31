#include "SparseMatrixDenseVectorMultiply.h"

/**
 * @brief Function to execute the sparse matrix-dense vector multiplication using sequential algorithm
 *
 * @param sparseMatrix  Sparse matrix
 * @param denseVector Dense vector
 * @param vecCols  Number of columns in the dense vector
 * @return DenseVector  Result of the multiplication
 */
DenseVector sparseMatrixDenseVectorMultiply(const SparseMatrix &sparseMatrix,
                                            const DenseVector &denseVector, int vecCols)
{
    int m = sparseMatrix.numRows; // Number of rows in the sparse matrix

    // Initialisation of the result vector
    DenseVector result(m, std::vector<double>(vecCols, 0.0));
    // Iterate over the rows of the sparse matrix
    for (int i = 0; i < m; ++i)
    {
        // Iterate over the non-zero elements in the current row
        for (int j = sparseMatrix.rowPtr[i]; j < sparseMatrix.rowPtr[i + 1]; ++j)
        {
            // Iterate over the columns of the dense vector
            for (int k = 0; k < vecCols; ++k)
            {
                result[i][k] += sparseMatrix.values[j] * denseVector[sparseMatrix.colIndices[j]][k]; // Compute the result
            }
        }
    }
    // Return the result
    return result;
}