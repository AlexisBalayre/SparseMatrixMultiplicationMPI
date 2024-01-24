#ifndef SPARSEMATRIXDENSEVECTORMULTIPLY_H
#define SPARSEMATRIXDENSEVECTORMULTIPLY_H

#include "MatrixDefinitions.h"

// Function to execute the sparse matrix-dense vector multiplication using sequential algorithm
DenseMatrix sparseMatrixDenseVectorMultiply(const SparseMatrix &sparseMatrix,
                                            const DenseMatrix &denseVector,
                                            int numRows, int numCols, int vecCols);

#endif
