#ifndef SPARSEMATRIXDENSEVECTORMULTIPLYNONZEROELEMENT_H
#define SPARSEMATRIXDENSEVECTORMULTIPLYNONZEROELEMENT_H

#include "MatrixDefinitions.h"

// Function to execute the sparse matrix-dense vector multiplication using non-zero element parallel algorithm
DenseMatrix sparseMatrixDenseVectorMultiplyNonZeroElement(const SparseMatrix &sparseMatrix, const DenseMatrix &denseVector);

#endif
