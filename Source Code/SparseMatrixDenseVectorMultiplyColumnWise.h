#ifndef SPARSEMATRIXDENSEVECTORMULTIPLYCOLUMNWIZE_H
#define SPARSEMATRIXDENSEVECTORMULTIPLYCOLUMNWIZE_H

#include "MatrixDefinitions.h"

// Function to execute the sparse matrix-dense vector multiplication using column-wise parallel algorithm
DenseMatrix sparseMatrixDenseVectorMultiplyColumnWise(const SparseMatrix &sparseMatrix, const DenseMatrix &denseVector, int numRows, int numCols, int vecCols);

#endif
