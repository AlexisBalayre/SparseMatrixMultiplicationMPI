#include "utils.h"                                         // Utility functions
#include "SparseMatrixDenseVectorMultiply.h"               // Sequential algorithm
#include "SparseMatrixDenseVectorMultiplyRowWise.h"        // Parallel algorithm (row-wise)
#include "SparseMatrixDenseVectorMultiplyColumnWise.h"     // Parallel algorithm (column-wise)
#include "SparseMatrixDenseVectorMultiplyNonZeroElement.h" // Parallel algorithm (non-zero element)

int main(int argc, char *argv[])
{
    // Initialise MPI and PETSc
    MPI_Init(&argc, &argv);
    PetscInitialize(&argc, &argv, NULL, NULL);

    // Retrieve the rank and size of the world communicator
    int worldRank, worldSize;
    MPI_Comm_rank(PETSC_COMM_WORLD, &worldRank);
    MPI_Comm_size(PETSC_COMM_WORLD, &worldSize);

    // Check if the correct number of arguments is provided
    if (argc != 3)
    {
        if (worldRank == 0)
        {
            std::cerr << "Usage: " << argv[0] << " <number of columns> <matrix file path>" << std::endl;
        }
        MPI_Abort(PETSC_COMM_WORLD, 1);
    }

    // Parse the command-line arguments
    int k = std::atoi(argv[1]);     // Convert the first argument to an integer
    std::string filename = argv[2]; // The second argument is the filename

    // Declare the sparse matrix and dense vector
    SparseMatrix M;
    DenseVector v;

    std::vector<double> flatData;
    int dataSize = 0;
    double startTime, endTime;

    // Declare the result of the serial multiplication
    DenseVector resultSerial;

    if (worldRank == 0)
    {
        std::cout << "World size: " << worldSize << std::endl;   // Print the number of processes
        std::cout << "Sparse matrix: " << filename << std::endl; // Print the name of the Matrix Market file

        // Read the sparse matrix from the Matrix Market file
        M = readMatrixMarketFile(filename);
        std::cout << "Matrix size: " << M.numRows << "x" << M.numCols << std::endl;

        // Generate a random dense vector
        v = generateLargeDenseVector(M.numCols, k);
        std::cout << "Vector size: " << M.numCols << "x" << k << std::endl;

        // Prepare the data for broadcasting
        flatData = serialize(v);    // Serialize the dense vector
        dataSize = flatData.size(); // Size of the serialized data

        // Execute the serial multiplication
        startTime = MPI_Wtime();
        resultSerial = sparseMatrixDenseVectorMultiply(M, v, k);
        endTime = MPI_Wtime();
        std::cout << "Serial Algo Execution time: " << (endTime - startTime)
                  << std::endl;

        // FOR DEBUGGING ONLY - PRINT 10 FIRST ELEMENTS OF THE RESULT
        // std::cout << "Result: " << std::endl;
        // for (int i = 0; i < 10; ++i)
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

    // Broadcast the Sparse Matrix to all processes
    // Prepare the data for broadcasting
    int valuesSize = M.values.size();                          // Number of non-zero elements
    int colIndicesSize = M.colIndices.size();                  // Number of column indices
    int rowPtrSize = M.rowPtr.size();                          // Number of row pointers
    MPI_Bcast(&M.numRows, 1, MPI_INT, 0, MPI_COMM_WORLD);      // Broadcast the number of rows
    MPI_Bcast(&M.numCols, 1, MPI_INT, 0, MPI_COMM_WORLD);      // Broadcast the number of columns
    MPI_Bcast(&valuesSize, 1, MPI_INT, 0, MPI_COMM_WORLD);     // Broadcast the number of non-zero elements
    MPI_Bcast(&colIndicesSize, 1, MPI_INT, 0, MPI_COMM_WORLD); // Broadcast the number of column indices
    MPI_Bcast(&rowPtrSize, 1, MPI_INT, 0, MPI_COMM_WORLD);     // Broadcast the number of row pointers
    // Resize the vectors for all processes
    if (worldRank != 0)
    {
        M.values.resize(valuesSize);
        M.colIndices.resize(colIndicesSize);
        M.rowPtr.resize(rowPtrSize);
    }
    // Broadcast the data
    MPI_Bcast(M.values.data(), valuesSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(M.colIndices.data(), colIndicesSize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(M.rowPtr.data(), rowPtrSize, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast the Dense Vector to all processes
    // Broadcast the size of the serialized data
    MPI_Bcast(&dataSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Resize flatData for all processes
    if (worldRank != 0)
    {
        flatData.resize(dataSize);
    }
    // Broadcast the data
    MPI_Bcast(flatData.data(), dataSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Deserialize the data
    if (worldRank != 0)
    {
        v.resize(M.numCols, std::vector<double>(k));
        v = deserialize(flatData, M.numCols, k);
    }

    // Wait for all processes to finish the broadcast
    MPI_Barrier(MPI_COMM_WORLD);

    // Execute the parallel multiplication (row-wise)
    startTime = MPI_Wtime();
    DenseVector resultRowWise = sparseMatrixDenseVectorMultiplyRowWise(M, v, k);
    endTime = MPI_Wtime();

    // Only the main process prints the parallel execution time
    if (worldRank == 0)
    {
        std::cout << "Row-wise Execution time: " << (endTime - startTime)
                  << std::endl;

        // FOR DEBUGGING ONLY - PRINT 10 FIRST ELEMENTS OF THE RESULT
        // std::cout << "Result: " << std::endl;
        // for (int i = 0; i < 10; ++i)
        // {
        //     for (int j = 0; j < k; ++j)
        //     {
        //         std::cout << resultRowWise[i][j] << " ";
        //     }
        //     std::cout << std::endl;
        // }

        // Compare the results of the serial and parallel multiplications
        if (areMatricesEqual(resultSerial, resultRowWise, 1e-6)) // Tolerance = 1e-6
        {
            std::cout << "Row-wise: Results are the same!"
                      << std::endl;
        }
        else
        {
            std::cout << "Row-wise: Results are different!"
                      << std::endl;
        }
    }

    // Wait for all processes to finish the parallel multiplication (row-wise)
    MPI_Barrier(MPI_COMM_WORLD);

    // Execute the parallel multiplication (column-wise)
    startTime = MPI_Wtime();
    DenseVector resultColumnWise = sparseMatrixDenseVectorMultiplyColumnWise(M, v, k);
    endTime = MPI_Wtime();

    // Only the main process prints the parallel execution time
    if (worldRank == 0)
    {
        std::cout << "Column-wise Execution time: " << (endTime - startTime)
                  << std::endl;

        // FOR DEBUGGING ONLY - PRINT 10 FIRST ELEMENTS OF THE RESULT
        // std::cout << "Result: " << std::endl;
        // for (int i = 0; i < 10; ++i)
        // {
        //     for (int j = 0; j < k; ++j)
        //     {
        //         std::cout << resultColumnWise[i][j] << " ";
        //     }
        //     std::cout << std::endl;
        // }

        // Compare the results of the serial and parallel multiplications
        if (areMatricesEqual(resultSerial, resultColumnWise, 1e-6)) // Tolerance = 1e-6
        {
            std::cout << "Column-wise: Results are the same!"
                      << std::endl;
        }
        else
        {
            std::cout << "Column-wise: Results are different!"
                      << std::endl;
        }
    }

    // Wait for all processes to finish the parallel multiplication (column-wise)
    MPI_Barrier(MPI_COMM_WORLD);

    // Execute the parallel multiplication (non-zero element)
    startTime = MPI_Wtime();
    DenseVector resultNonZeroElement = sparseMatrixDenseVectorMultiplyNonZeroElement(M, v, k);
    endTime = MPI_Wtime();

    // Only the main process prints the parallel execution time
    if (worldRank == 0)
    {
        std::cout << "Non-zero Elements Execution time: " << (endTime - startTime)
                  << std::endl;

        // FOR DEBUGGING ONLY - PRINT 10 FIRST ELEMENTS OF THE RESULT
        // std::cout << "Result: " << std::endl;
        // for (int i = 0; i < 10; ++i)
        // {
        //     for (int j = 0; j < k; ++j)
        //     {
        //         std::cout << resultNonZeroElement[i][j] << " ";
        //     }
        //     std::cout << std::endl;
        // }

        // Compare the results of the serial and parallel multiplications
        if (areMatricesEqual(resultSerial, resultNonZeroElement, 1e-6)) // Tolerance = 1e-6
        {
            std::cout << "Non-zero Elements: Results are the same!"
                      << std::endl;
        }
        else
        {
            std::cout << "Non-zero Elements: Results are different!"
                      << std::endl;
        }
    }

    // Wait for all processes to finish the parallel multiplication (non-zero element)
    MPI_Barrier(MPI_COMM_WORLD);

    // Declare the PETSc matrix
    Mat A, B, C;

    // Create a parallel matrix to store the sparse matrix
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, M.numRows, M.numCols);
    MatSetType(A, MATMPIAIJ);
    MatSetUp(A);
    // Fill the PETSc matrix with the values from the sparse matrix
    if (worldRank == 0)
    {
        for (int i = 0; i < M.numRows; ++i)
        {
            for (int j = M.rowPtr[i]; j < M.rowPtr[i + 1]; ++j)
            {
                MatSetValue(A, i, M.colIndices[j], M.values[j], INSERT_VALUES);
            }
        }
    }
    // Assemble the PETSc matrix
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    // Create a parallel matrix to store the dense vector
    MatCreate(PETSC_COMM_WORLD, &B);
    MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, M.numCols, k);
    MatSetType(B, MATDENSE);
    MatSetUp(B);
    // Fill the PETSc matrix B with values from the dense matrix v
    if (worldRank == 0)
    {
        for (int i = 0; i < M.numCols; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                MatSetValue(B, i, j, v[i][j], INSERT_VALUES);
            }
        }
    }
    // Assemble the PETSc matrix
    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);

    // Create a parallel matrix to store the result of the multiplication
    MatProductCreate(A, B, NULL, &C);

    startTime = MPI_Wtime();
    MatMatMult(A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);
    endTime = MPI_Wtime();

    // Create a sequential matrix to retrieve the result
    Mat CSeq;
    MatCreateRedundantMatrix(C, worldSize, MPI_COMM_NULL, MAT_INITIAL_MATRIX, &CSeq);

    if (worldRank == 0)
    {
        // Print the execution time
        std::cout << "PETSc Execution time: " << (endTime - startTime) << std::endl;

        // Convert the result matrix C to a DenseVector
        DenseVector globalMatrix = ConvertPETScMatToDenseVector(CSeq);

        // FOR DEBUGGING ONLY - PRINT 10 FIRST ELEMENTS OF THE RESULT
        // std::cout << "Result: " << std::endl;
        // for (int i = 0; i < 10; ++i)
        // {
        //     for (int j = 0; j < k; ++j)
        //     {
        //         std::cout << globalMatrix[i][j] << " ";
        //     }
        //     std::cout << std::endl;
        // }

        // Compare the results of the serial and PETSc multiplications
        if (areMatricesEqual(resultSerial, globalMatrix, 1e-6)) // Tolerance = 1e-6
        {
            std::cout << "PETSc: Results are the same!"
                      << std::endl;
        }
        else
        {
            std::cout << "PETSc: Results are different!"
                      << std::endl;
        }
    }

    // Free the memory
    MatDestroy(&A);
    MatDestroy(&B);
    MatDestroy(&C);
    MatDestroy(&CSeq);

    // Finalise MPI and PETSc
    PetscFinalize();
    MPI_Finalize();

    return 0;
}