#include <iostream>
#include <vector>
#include <cstdlib> // For rand() and srand()
#include <ctime>   // For time()
#include <chrono>
#include <mpi.h>
#include <petsc.h>
#include <fstream>
#include <sstream>
#include <utility>   // Pour std::pair
#include <algorithm> // Pour std::sort
#include <stdexcept>
#include <string>
#include "SparseMatrixDenseVectorMultiply.h" // Sequential algorithm
#include <cmath>                             // Pour std::fabs

DenseVector ConvertPETScMatToDenseVector(Mat C)
{
    PetscInt m, n;
    MatGetSize(C, &m, &n);

    DenseVector denseVec(m, std::vector<double>(n, 0.0));

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; j++)
        {
            PetscScalar value;
            MatGetValue(C, i, j, &value);

            // Value should be real
            denseVec[i][j] = PetscRealPart(value);
        }
    }

    return denseVec;
}

bool areMatricesEqual(const DenseVector &mat1, const DenseVector &mat2, double tolerance)
{
    if (mat1.size() != mat2.size())
        return false;

    for (size_t i = 0; i < mat1.size(); ++i)
    {
        if (mat1[i].size() != mat2[i].size())
            return false;

        for (size_t j = 0; j < mat1[i].size(); ++j)
        {
            if (std::fabs(mat1[i][j] - mat2[i][j]) > tolerance)
            {
                return false;
            }
        }
    }

    return true;
}

SparseMatrix readMatrixMarketFile(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    std::string line;
    bool isSymmetric = false;
    while (std::getline(file, line))
    {
        if (line[0] == '%')
        {
            if (line.find("symmetric") != std::string::npos)
            {
                isSymmetric = true;
            }
        }
        else
        {
            break; // First non-comment line reached, break out of the loop
        }
    }

    int numRows, numCols, nonZeros;
    std::stringstream(line) >> numRows >> numCols >> nonZeros;
    if (!file)
    {
        throw std::runtime_error("Failed to read matrix dimensions from file: " + filename);
    }

    SparseMatrix matrix;
    matrix.rowPtr.resize(numRows + 1, 0);
    std::vector<std::vector<std::pair<int, double>>> tempRows(numRows);

    int rowIndex, colIndex;
    double value;
    for (int i = 0; i < nonZeros; ++i)
    {
        file >> rowIndex >> colIndex >> value;
        if (!file)
        {
            throw std::runtime_error("Failed to read data from file: " + filename);
        }

        rowIndex--; // Adjusting from 1-based to 0-based indexing
        colIndex--;

        tempRows[rowIndex].emplace_back(colIndex, value);

        if (isSymmetric && rowIndex != colIndex)
        {
            tempRows[colIndex].emplace_back(rowIndex, value);
        }
    }

    // Trier chaque ligne par indice de colonne
    for (auto &row : tempRows)
    {
        std::sort(row.begin(), row.end());
    }

    // Reconstruire la structure SparseMatrix
    int cumSum = 0;
    for (int i = 0; i < numRows; ++i)
    {
        matrix.rowPtr[i] = cumSum;
        for (const auto &elem : tempRows[i])
        {
            matrix.values.push_back(elem.second);
            matrix.colIndices.push_back(elem.first);
        }
        cumSum += tempRows[i].size();
    }
    matrix.rowPtr[numRows] = cumSum;

    matrix.numRows = numRows;
    matrix.numCols = numCols;

    return matrix;
}

// Function to generate a large dense vector
DenseVector generateLargeDenseVector(int n, int k)
{
    DenseVector matrix(n, std::vector<double>(k));

    // Fill the matrix with random values
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            matrix[i][j] = rand() % 100 + 1; // Random value between 1 and 100
        }
    }

    return matrix;
}

int main(int argc, char *argv[])
{
    // Initialize MPI and PETSc
    MPI_Init(&argc, &argv);
    PetscInitialize(&argc, &argv, NULL, NULL);

    int worldRank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &worldRank);

    int worldSize;
    MPI_Comm_size(PETSC_COMM_WORLD, &worldSize);

    int k = 2; // Number of columns in the dense vector

    // Seul le processus principal génère la matrice et le vecteur
    srand(time(0));

    SparseMatrix M;
    DenseVector v;
    DenseVector resultSerial;

    int m, n;

    if (worldRank == 0)
    {
        M = readMatrixMarketFile("./sparse-matrix/cop20k_A.mtx");
        m = M.numRows;
        n = M.numCols;
        std::cout << "Matrix size: " << m << "x" << n << std::endl;
        v = generateLargeDenseVector(n, k);
        std::cout << "Vector size: " << n << "x" << k << std::endl;
        auto startSerial = std::chrono::high_resolution_clock::now();
        resultSerial = sparseMatrixDenseVectorMultiply(M, v, k);
        auto stopSerial = std::chrono::high_resolution_clock::now();
        auto durationSerial = std::chrono::duration_cast<std::chrono::milliseconds>(stopSerial - startSerial);
        std::cout << "Serial Execution time: " << durationSerial.count() << " milliseconds" << std::endl;

        // FOR DEBUGGING ONLY
        // std::cout << "Result: " << std::endl;
        // for (int i = 0; i < m; ++i)
        // {
        //     for (int j = 0; j < k; ++j)
        //     {
        //         std::cout << resultSerial[i][j] << " ";
        //     }
        //     std::cout << std::endl;
        // }
    }

    // Wait for the main process to finish the serial multiplication
    MPI_Barrier(MPI_COMM_WORLD);

    // Broadcast m and n to all processes
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    PetscBarrier(NULL);

    // PETSc Matrix and Vector setup for Sparse Matrix
    Mat A;
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m, n);
    MatSetType(A, MATMPIAIJ);
    MatSetUp(A);

    // Fill the PETSc matrix with the values from the sparse matrix
    if (worldRank == 0)
    {
        for (int i = 0; i < m; ++i)
        {
            for (int j = M.rowPtr[i]; j < M.rowPtr[i + 1]; ++j)
            {
                MatSetValue(A, i, M.colIndices[j], M.values[j], INSERT_VALUES);
            }
        }
    }

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    // PETSc Matrix setup for Dense Matrix
    Mat B, C;

    // Create a parallel dense matrix
    MatCreate(PETSC_COMM_WORLD, &B);
    MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, n, k);
    MatSetType(B, MATDENSE);
    MatSetUp(B);

    // Fill the PETSc matrix B with values from the dense matrix v
    if (worldRank == 0)
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                MatSetValue(B, i, j, v[i][j], INSERT_VALUES);
            }
        }
    }

    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);

    // Create matrix C to store the result
    MatProductCreate(A, B, NULL, &C);

    // Perform the multiplication
    auto startPETSc = std::chrono::high_resolution_clock::now();
    MatMatMult(A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);
    auto stopPETSc = std::chrono::high_resolution_clock::now();
    auto durationPETSc = std::chrono::duration_cast<std::chrono::milliseconds>(stopPETSc - startPETSc);

    if (worldRank == 0)
    {
        std::cout << "PETSc Execution time: " << durationPETSc.count() << " milliseconds" << std::endl;
    }

    // Create a sequential dense matrix to store the result
    Mat CSeq;
    MatCreateRedundantMatrix(C, worldSize, MPI_COMM_NULL, MAT_INITIAL_MATRIX, &CSeq);

    // Compare matrices on process 0
    if (worldRank == 0)
    {

        // Convert the result matrix C to a DenseVector
        DenseVector globalMatrix = ConvertPETScMatToDenseVector(CSeq);

        /* // print size of the matrix
        std::cout << "Matrix size: " << resultSerial.size() << "x" << resultSerial[0].size() << std::endl;
        // Print 10 first rows
        std::cout << "First 10 rows: " << std::endl;
        for (int i = 0; i < 10; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                std::cout << resultSerial[i][j] << " ";
            }
            std::cout << std::endl;
        } */

        // print size of the matrix
        /* std::cout << "Matrix size: " << globalMatrix.size() << "x" << globalMatrix[0].size() << std::endl;
        // Print 10 first rows
        std::cout << "First 10 rows: " << std::endl;
        for (int i = 0; i < 10; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                std::cout << globalMatrix[i][j] << " ";
            }
            std::cout << std::endl;
        } */

        if (areMatricesEqual(resultSerial, globalMatrix, 1e-6))
        {
            std::cout << "Results are the same." << std::endl;
        }
        else
        {
            std::cout << "Results are different!" << std::endl;
        }
    }

    // Wait for all processes to finish
    PetscBarrier(NULL);

    MatDestroy(&A);
    MatDestroy(&B);
    MatDestroy(&C);
    MatDestroy(&CSeq);

    PetscFinalize();

    MPI_Finalize();
    return 0;
}