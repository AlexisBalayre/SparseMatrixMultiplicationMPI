#ifndef UTILS_H
#define UTILS_H

#include <iostream> // std::cout
#include <vector>   // std::vector
#include <cstdlib>  // rand() and srand()
#include <ctime>    // time()
#include <mpi.h>
#include <petsc.h>
#include <fstream>   // std::ifstream
#include <string>    // std::string
#include <sstream>   // std::stringstream
#include <utility>   // std::pair
#include <algorithm> // std::sort
#include <stdexcept> // std::runtime_error
#include <cmath>     // std::fabs
#include "MatrixDefinitions.h"

/**
 * Method to convert a PETSc matrix to a fat vector
 * @param C PETSc matrix
 * @return FatVector Dense vector
 */
FatVector ConvertPETScMatToFatVector(Mat C);

/**
 * Method to compare two matrices
 * @param mat1 First matrix
 * @param mat2 Second matrix
 * @param tolerance Tolerance for comparison
 * @return bool True if the matrices are equal, false otherwise
 */
bool areMatricesEqual(const FatVector &mat1, const FatVector &mat2, double tolerance);

/**
 * Method to read a matrix from a Matrix Market file
 * @param filename Name of the file
 * @return SparseMatrix Sparse matrix
 */
SparseMatrix readMatrixMarketFile(const std::string &filename);

/**
 * Method to generate a random fat vector
 * @param n Number of rows
 * @param m Number of columns
 * @return FatVector Dense vector
 */
FatVector generateLargeFatVector(int n, int k);

/**
 * @brief Method to serialize a FatVector to a flat array
 * @param denseVec Dense vector to serialize
 * @return std::vector<double> Flat array containing the serialized data
 */
std::vector<double> serialize(const FatVector &denseVec);

/**
 * @brief Method to deserialize a flat array to a FatVector
 * @param flat Flat array to deserialize
 * @param rows Number of rows in the fat vector
 * @param cols Number of columns in the fat vector
 * @return FatVector Dense vector
 */
FatVector deserialize(const std::vector<double> &flat, int rows, int cols);

#endif
