#include "SparseMatrixFatVectorMultiply.h"

/**
 * @brief Function to execute the sparse matrix-Fat Vector multiplication using sequential algorithm
 *
 * @param sparseMatrix  Sparse matrix
 * @param fatVector Fat Vector
 * @param vecCols  Number of columns in the Fat Vector
 * @return FatVector  Result of the multiplication
 */
FatVector sparseMatrixFatVectorMultiply(const SparseMatrix &sparseMatrix,
                                            const FatVector &fatVector, int vecCols)
{
    // Initialisation of the result vector
    FatVector result(sparseMatrix.numRows, std::vector<double>(vecCols, 0.0));
    // Iterate over the rows of the sparse matrix
    for (int i = 0; i < sparseMatrix.numRows; ++i)
    {
        // Iterate over the non-zero elements in the current row
        for (int j = sparseMatrix.rowPtr[i]; j < sparseMatrix.rowPtr[i + 1]; ++j)
        {
            // Iterate over the columns of the Fat Vector
            for (int k = 0; k < vecCols; ++k)
            {
                result[i][k] += sparseMatrix.values[j] * fatVector[sparseMatrix.colIndices[j]][k]; // Compute the result
            }
        }
    }
    // Return the result
    return result;
}