#include "utils.h"

/**
 * Method to convert a PETSc matrix to a dense vector
 * @param C PETSc matrix
 * @return DenseVector Dense vector
 */
DenseVector ConvertPETScMatToDenseVector(Mat C)
{
    PetscInt m, n;         // Number of rows and columns in the matrix
    MatGetSize(C, &m, &n); // Get the number of rows and columns in the matrix

    DenseVector denseVec(m, std::vector<double>(n, 0.0)); // Dense vector to hold the matrix

    // Iterate over the rows of the matrix
    for (int i = 0; i < m; ++i)
    {
        // Iterate over the columns of the matrix
        for (int j = 0; j < n; j++)
        {
            PetscScalar value;                     // Value of the element
            MatGetValue(C, i, j, &value);          // Get the value of the element
            denseVec[i][j] = PetscRealPart(value); // Copy the value of the element
        }
    }

    // Return the dense vector
    return denseVec;
}

/**
 * Method to compare two matrices
 * @param mat1 First matrix
 * @param mat2 Second matrix
 * @param tolerance Tolerance for comparison
 * @return bool True if the matrices are equal, false otherwise
 */
bool areMatricesEqual(const DenseVector &mat1, const DenseVector &mat2, double tolerance)
{
    // Check if the matrices have the same dimensions
    if (mat1.size() != mat2.size())
        return false;

    // Iterate over the rows of the matrices
    for (size_t i = 0; i < mat1.size(); ++i)
    {
        // Check if the rows have the same dimensions
        if (mat1[i].size() != mat2[i].size())
            return false;

        // Iterate over the columns of the matrices
        for (size_t j = 0; j < mat1[i].size(); ++j)
        {
            // Check if the elements are equal
            if (std::fabs(mat1[i][j] - mat2[i][j]) > tolerance)
            {
                return false; // Matrices are not equal
            }
        }
    }

    return true; // Matrices are equal
}

/**
 * Method to read a matrix from a Matrix Market file
 * @param filename Name of the file
 * @return SparseMatrix Sparse matrix
 */
SparseMatrix readMatrixMarketFile(const std::string &filename)
{
    std::ifstream file(filename); // Input file stream

    // Check if the file was opened successfully
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    std::string line;                            // String to hold the current line
    bool isSymmetric = false, isPattern = false; // Flags to indicate if the matrix is symmetric or pattern only

    // Skip the comments
    while (std::getline(file, line))
    {
        // Check if the line is a comment
        if (line[0] == '%')
        {
            // Check if the line contains the word "symmetric"
            if (line.find("symmetric") != std::string::npos)
            {
                isSymmetric = true; // Set the symmetric flag
            }

            // Check if the line contains the word "pattern"
            if (line.find("pattern") != std::string::npos)
            {
                isPattern = true; // Set the pattern flag
            }
        }
        else
        {
            break; // First non-comment line reached, break out of the loop
        }
    }

    // Read the matrix dimensions
    int numRows, numCols, nonZeros;                            // Number of rows, columns and non-zero elements in the matrix
    std::stringstream(line) >> numRows >> numCols >> nonZeros; // Read the dimensions from the line

    // Check if the file was read successfully
    if (!file)
    {
        throw std::runtime_error("Failed to read matrix dimensions from file: " + filename);
    }

    SparseMatrix matrix;                                                // Sparse matrix to hold the data
    matrix.rowPtr.resize(numRows + 1, 0);                               // Resize the row pointer vector
    std::vector<std::vector<std::pair<int, double>>> tempRows(numRows); // Temporary vector to hold the data
    int rowIndex, colIndex;                                             // Row and column indices
    double value;                                                       // Value of the non-zero element

    // Read the non-zero elements
    for (int i = 0; i < nonZeros; ++i)
    {
        // If the matrix is pattern only, the value of the non-zero element is 1.0
        if (isPattern)
        {
            file >> rowIndex >> colIndex; // Read the row and column indices
            value = 1.0;                  // Default value for pattern entries
        }
        else
        {
            file >> rowIndex >> colIndex >> value; // Read the row and column indices and the value
        }

        // Check if the file was read successfully
        if (!file)
        {
            throw std::runtime_error("Failed to read data from file: " + filename);
        }

        rowIndex--; // Adjusting from 1-based to 0-based indexing
        colIndex--; // Adjusting from 1-based to 0-based indexing

        tempRows[rowIndex].emplace_back(colIndex, value); // Store the data in the temporary vector

        // If the matrix is symmetric, store the data in the transpose as well
        if (isSymmetric && rowIndex != colIndex)
        {
            tempRows[colIndex].emplace_back(rowIndex, value); // Store the data in the temporary vector
        }
    }

    // Sort each row by column index
    for (auto &row : tempRows)
    {
        std::sort(row.begin(), row.end());
    }

    // Reconstruct SparseMatrix structure
    int cumSum = 0; // Cumulative sum of the number of non-zero elements

    // Iterate over the rows of the matrix
    for (int i = 0; i < numRows; ++i)
    {
        matrix.rowPtr[i] = cumSum; // Store the cumulative sum in the row pointer vector

        // Iterate over the non-zero elements in the current row
        for (const auto &elem : tempRows[i])
        {
            matrix.values.push_back(elem.second);    // Store the value of the non-zero element
            matrix.colIndices.push_back(elem.first); // Store the column index of the non-zero element
        }

        cumSum += tempRows[i].size(); // Update the cumulative sum
    }

    matrix.rowPtr[numRows] = cumSum; // Store the cumulative sum in the row pointer vector
    matrix.numRows = numRows;        // Store the number of rows
    matrix.numCols = numCols;        // Store the number of columns

    // Return the sparse matrix
    return matrix;
}

/**
 * Method to generate a random dense vector
 * @param n Number of rows
 * @param m Number of columns
 * @return DenseVector Dense vector
 */
DenseVector generateLargeDenseVector(int n, int k)
{
    DenseVector denseVector(n, std::vector<double>(k)); // Dense vector to hold the random values

    // Iterate over the rows of the dense vector
    for (int i = 0; i < n; ++i)
    {
        // Iterate over the columns of the dense vector
        for (int j = 0; j < k; ++j)
        {
            denseVector[i][j] = rand() % 100 + 1; // Generate a random value between 1 and 100
        }
    }

    // Return the dense vector
    return denseVector;
}

/**
 * @brief Method to serialize a DenseVector to a flat array
 * @param denseVec Dense vector to serialize
 * @return std::vector<double> Flat array containing the serialized data
 */
std::vector<double> serialize(const DenseVector &denseVec)
{
    std::vector<double> flat; // Flat array to hold the serialized data

    // Iterate over the rows of the dense vector
    for (const auto &vec : denseVec)
    {
        flat.insert(flat.end(), vec.begin(), vec.end()); // Copy the elements
    }

    // Return the flat array
    return flat;
}

/**
 * @brief Method to deserialize a flat array to a DenseVector
 * @param flat Flat array to deserialize
 * @param rows Number of rows in the dense vector
 * @param cols Number of columns in the dense vector
 * @return DenseVector Dense vector
 */
DenseVector deserialize(const std::vector<double> &flat, int rows, int cols)
{
    DenseVector denseVec(rows, std::vector<double>(cols)); // Dense vector to hold the deserialized data

    // Iterate over the rows of the dense vector
    for (int i = 0; i < rows; ++i)
    {
        // Iterate over the columns of the dense vector
        for (int j = 0; j < cols; ++j)
        {
            denseVec[i][j] = flat[i * cols + j]; // Copy the element
        }
    }

    // Return the dense vector
    return denseVec;
}