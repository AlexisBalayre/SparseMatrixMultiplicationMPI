#ifndef MATRIXDEFINITIONS_H
#define MATRIXDEFINITIONS_H

#include <vector>

//
/**
 * @brief Struct to represent a sparse matrix
 *
 * @param values  Non-zero values
 * @param colIndices  Column indices of non-zero values
 * @param rowPtr  Row pointers
 */
struct SparseMatrix
{
    std::vector<double> values;
    std::vector<int> colIndices;
    std::vector<int> rowPtr;
};

// Type definition for a dense vector
typedef std::vector<std::vector<double>> DenseVector;

#endif
