#include <iostream>
#include <vector>
#include <cstdlib> // For rand() and srand()
#include <ctime>   // For time()
#include <chrono>
#include <mpi.h>
#include <algorithm>                                       // For std::sort()
#include "SparseMatrixDenseVectorMultiply.h"               // Sequential algorithm
#include "SparseMatrixDenseVectorMultiplyRowWise.h"        // Parallel algorithm (row-wise)
#include "SparseMatrixDenseVectorMultiplyColumnWise.h"     // Parallel algorithm (column-wise)
#include "SparseMatrixDenseVectorMultiplyNonZeroElement.h" // Parallel algorithm (non-zero element)

// Function to generate a large sparse matrix
SparseMatrix generateLargeSparseMatrix(int m, int n, double nonZeroPercentage)
{
    SparseMatrix matrix;
    matrix.rowPtr.resize(m + 1);

    int totalElements = m * n;
    int nonZeroElements = static_cast<int>(totalElements * nonZeroPercentage / 100);

    // Reserving space for efficiency
    matrix.values.reserve(nonZeroElements);
    matrix.colIndices.reserve(nonZeroElements);

    // Evenly distributing non-zero elements across rows
    for (int i = 0; i < m; ++i)
    {
        int nonZerosInRow = nonZeroElements / m + (i < nonZeroElements % m);
        for (int j = 0; j < nonZerosInRow; ++j)
        {
            matrix.values.push_back(rand() % 100 + 1); // Random value between 1 and 100
            matrix.colIndices.push_back(rand() % n);   // Random column index
        }
    }

    // Sorting each row's column indices to maintain CSR format and avoid duplicates
    int startIdx = 0;
    for (int i = 0; i < m; ++i)
    {
        matrix.rowPtr[i] = startIdx;
        int nonZerosInRow = nonZeroElements / m + (i < nonZeroElements % m);
        std::sort(matrix.colIndices.begin() + startIdx, matrix.colIndices.begin() + startIdx + nonZerosInRow);
        startIdx += nonZerosInRow;
    }
    matrix.rowPtr[m] = nonZeroElements;

    return matrix;
}

// Function to generate a large dense vector
DenseVector generateLargeDenseVector(int m, int n)
{
    DenseVector matrix(m, std::vector<double>(n));

    // Fill the matrix with random values
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            matrix[i][j] = rand() % 100 + 1; // Random value between 1 and 100
        }
    }

    return matrix;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int worldRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    double nonZeroPercentage = 1.0;
    int m = 10000;
    int n = 5000;
    int k = 10;

    // Seul le processus principal génère la matrice et le vecteur
    srand(time(0)); // Seed pour la génération aléatoire
    SparseMatrix M = generateLargeSparseMatrix(m, n, nonZeroPercentage);
    DenseVector v = generateLargeDenseVector(n, k);

    // Serial multiplication (only in the main process)
    DenseVector resultSerial;
    if (worldRank == 0)
    {
        auto startSerial = std::chrono::high_resolution_clock::now();
        resultSerial = sparseMatrixDenseVectorMultiply(M, v, m, n, k);
        auto stopSerial = std::chrono::high_resolution_clock::now();
        auto durationSerial = std::chrono::duration_cast<std::chrono::milliseconds>(stopSerial - startSerial);
        std::cout << "Serial Execution time: " << durationSerial.count() << " milliseconds" << std::endl;
        /*  std::cout << "Result: " << std::endl;
         for (int i = 0; i < m; ++i)
         {
             for (int j = 0; j < k; ++j)
             {
                 std::cout << resultSerial[i][j] << " ";
             }
             std::cout << std::endl;
         }
          */
    }

    // Parallel multiplication (row-wise)
    auto startParallel = std::chrono::high_resolution_clock::now();
    DenseVector resultParallelRowWise = sparseMatrixDenseVectorMultiplyRowWise(M, v, m, n, k);
    auto stopParallel = std::chrono::high_resolution_clock::now();
    auto durationParallel = std::chrono::duration_cast<std::chrono::milliseconds>(stopParallel - startParallel);

    // Only the main process prints the parallel execution time
    if (worldRank == 0)
    {
        std::cout << "Parallel Execution time (row-wise): " << durationParallel.count() << " milliseconds" << std::endl;
        /* std::cout << "Result: " << std::endl;
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                std::cout << resultParallel[i][j] << " ";
            }
            std::cout << std::endl;
        } */

        // Verify that the results are the same: Model: Serial
        bool sameResult = true;
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                if (resultParallelRowWise[i][j] != resultSerial[i][j])
                {
                    sameResult = false;
                    break;
                }
            }
        }
        std::cout << "Same result: " << sameResult << std::endl;
    }

    // Parallel multiplication (column-wise)
    auto startParallelColumnWise = std::chrono::high_resolution_clock::now();
    DenseVector resultParallelColumnWise = sparseMatrixDenseVectorMultiplyColumnWise(M, v, m, n, k);
    auto stopParallelColumnWise = std::chrono::high_resolution_clock::now();
    auto durationParallelColumnWise = std::chrono::duration_cast<std::chrono::milliseconds>(stopParallelColumnWise - startParallelColumnWise);

    // Only the main process prints the parallel execution time
    if (worldRank == 0)
    {
        std::cout << "Parallel Execution time (column-wise): " << durationParallelColumnWise.count() << " milliseconds" << std::endl;
        /* std::cout << "Result: " << std::endl;
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                std::cout << resultParallelColumnWise[i][j] << " ";
            }
            std::cout << std::endl; */

        // Verify that the results are the same: Model: Serial
        bool sameResult = true;
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                if (resultParallelColumnWise[i][j] != resultSerial[i][j])
                {
                    sameResult = false;
                    break;
                }
            }
        }
        std::cout << "Same result: " << sameResult << std::endl;
    }

    // Parallel multiplication (non-zero element)
    auto startParallelNonZeroElement = std::chrono::high_resolution_clock::now();
    DenseVector resultParallelNonZeroElement = sparseMatrixDenseVectorMultiplyNonZeroElement(M, v, m, n, k);
    auto stopParallelNonZeroElement = std::chrono::high_resolution_clock::now();
    auto durationParallelNonZeroElement = std::chrono::duration_cast<std::chrono::milliseconds>(stopParallelNonZeroElement - startParallelNonZeroElement);

    // Only the main process prints the parallel execution time
    if (worldRank == 0)
    {
        std::cout << "Parallel Execution time (non-zero element): " << durationParallelNonZeroElement.count() << " milliseconds" << std::endl;
        /* std::cout << "Result: " << std::endl;
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                std::cout << resultParallelNonZeroElement[i][j] << " ";
            }
            std::cout << std::endl;
        }  */

        // Verify that the results are the same: Model: Serial
        bool sameResult = true;
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                if (resultParallelNonZeroElement[i][j] != resultSerial[i][j])
                {
                    sameResult = false;
                    break;
                }
            }
        }
        std::cout << "Same result: " << sameResult << std::endl;
    }

    MPI_Finalize();
    return 0;
}