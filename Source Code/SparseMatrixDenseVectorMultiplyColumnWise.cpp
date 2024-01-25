#include <mpi.h>
#include "SparseMatrixDenseVectorMultiplyColumnWise.h"

/**
 * @brief Function to execute the sparse matrix-dense vector multiplication using column-wise parallel algorithm
 *
 * @param sparseMatrix Sparse matrix
 * @param denseVector Dense vector
 * @param numRow Number of rows in the sparse matrix
 * @param numCols Number of columns in the sparse matrix
 * @param vecCols Number of columns in the dense vector
 * @return DenseVector Result of the multiplication
 */
DenseVector sparseMatrixDenseVectorMultiplyColumnWise(const SparseMatrix &sparseMatrix, const DenseVector &denseVector, int numRows, int numCols, int vecCols)
{
    // MPI Initialisation
    int worldSize, worldRank;                  // Total number of processes and the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize); // Get the total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank); // Get the rank of the current process

    int colsPerProcess = vecCols / worldSize; // Number of columns to be processed by each process
    int extraCols = vecCols % worldSize;      // Number of processes that will process one extra column
    int startCol, endCol;                     // Starting and ending column indices for the current process

    // Determine the starting and ending columns for the current process
    if (worldRank < extraCols)
    {
        startCol = worldRank * (colsPerProcess + 1);
        endCol = startCol + colsPerProcess + 1;
    }
    else
    {
        startCol = worldRank * colsPerProcess + extraCols;
        endCol = startCol + colsPerProcess;
    }

    // Local Result Vector Initialisation
    DenseVector localResult(numRows, std::vector<double>(endCol - startCol, 0.0));

    // Local computation: each process computes its portion of the result
    // Iterate over the columns assigned to the current process
    for (int col = startCol; col < endCol; ++col)
    {
        // Iterate over the rows of the sparse matrix
        for (int i = 0; i < numRows; ++i)
        {
            double sum = 0.0; // Variable to store the sum of the products
            // Iterate over the non-zero elements in the current row
            for (int j = sparseMatrix.rowPtr[i]; j < sparseMatrix.rowPtr[i + 1]; ++j)
            {
                int sparseCol = sparseMatrix.colIndices[j]; // Column index of the non-zero element
                // Check if the column index is within the range of the dense vector
                if (sparseCol < numCols)
                {
                    sum += sparseMatrix.values[j] * denseVector[sparseCol][col]; // Compute the sum of the products
                }
            }
            localResult[i][col - startCol] = sum; // Store the sum in the local result vector
        }
    }

    // Gather the local results in the root process
    std::vector<double> flatLocalResult;                    // Flat vector to store the local result
    flatLocalResult.reserve(numRows * (endCol - startCol)); // Reserve space for the flat vector
    // Iterate over the rows of the local result
    for (const auto &row : localResult)
    {
        flatLocalResult.insert(flatLocalResult.end(), row.begin(), row.end()); // Insert the row in the flat vector
    }

    int sendCount = flatLocalResult.size(); // Number of elements to be sent by each process

    std::vector<double> gatheredResults;                                 // Flat vector to store the gathered results
    std::vector<int> receiveCounts(worldSize), displacements(worldSize); // receiveCounts stores the number of elements to be received from each process, displacements stores the displacements for the gathered results

    // Prepare the receiveCounts and displacements vectors in the root process
    if (worldRank == 0)
    {
        gatheredResults.resize(numRows * vecCols); // Resize the gatheredResults vector to accommodate all results

        int displacement = 0; // Displacement for the current process
        for (int i = 0; i < worldSize; ++i)
        {
            int numColsProcessed = (i < extraCols) ? (colsPerProcess + 1) : colsPerProcess; // Number of columns processed by the current process
            receiveCounts[i] = numRows * numColsProcessed;                                  // Number of elements to be received from the current process
            displacements[i] = displacement;                                                // Displacement for the current process
            displacement += receiveCounts[i];                                               // Update the displacement
        }
    }

    // Gather all local results into the root process
    MPI_Gatherv(flatLocalResult.data(), sendCount, MPI_DOUBLE,
                gatheredResults.data(), receiveCounts.data(), displacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Reconstruct the final result matrix in the root process
    DenseVector finalResult; // Final result matrix
    if (worldRank == 0)
    {
        finalResult.resize(numRows, std::vector<double>(vecCols, 0.0)); // Resize the final result matrix to accommodate the final result
        int resultIndex = 0;                                            // Index of the current element in the gatheredResults vector

        // Iterate over the columns of the final result matrix
        for (int rank = 0; rank < worldSize; ++rank)
        {
            int numColsThisRank = (rank < extraCols) ? (colsPerProcess + 1) : colsPerProcess; // Number of columns processed by the current process
            int startColThisRank = rank * colsPerProcess + std::min(rank, extraCols);         // Starting column index for the current process

            // Iterate over the rows of the final result matrix
            for (int row = 0; row < numRows; ++row)
            {
                // Iterate over the columns of the final result matrix
                for (int col = 0; col < numColsThisRank; ++col)
                {
                    finalResult[row][startColThisRank + col] = gatheredResults[resultIndex++]; // Reconstruct the final result matrix
                }
            }
        }
    }

    return (worldRank == 0) ? finalResult : DenseVector{}; // Return the final result
}
