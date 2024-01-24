#ifndef MATRIXDEFINITIONS_H
#define MATRIXDEFINITIONS_H

#include <vector>

struct SparseMatrix {
    std::vector<double> values;       // Non-zero values
    std::vector<int> colIndices;      // Column indices of non-zero values
    std::vector<int> rowPtr;          // Row pointers
};


typedef std::vector<std::vector<double>> DenseMatrix;

#endif
