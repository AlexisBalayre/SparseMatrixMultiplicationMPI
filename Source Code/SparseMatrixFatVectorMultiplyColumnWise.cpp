#include <mpi.h>
#include "SparseMatrixFatVectorMultiplyColumnWise.h"
#include <numeric> // std::accumulate

/**
 * @brief Function to execute the sparse matrix-Fat Vector multiplication using column-wise parallel algorithm
 *
 * @param sparseMatrix Sparse matrix
 * @param fatVector Fat Vector
 * @param vecCols Number of columns in the Fat Vector
 * @return FatVector Result of the multiplication
 */
FatVector sparseMatrixFatVectorMultiplyColumnWise(const SparseMatrix &sparseMatrix, const FatVector &fatVector, int vecCols)
{
    // Retrieve the rank and size of the MPI world
    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    // =========================== FOR DEBUGGING ONLY - START LOCAL COMPUTATION TIMER ===============================
    // double computation_start = MPI_Wtime();
    // =========================== FOR DEBUGGING ONLY - START LOCAL COMPUTATION TIMER ===============================

    // Distribute columns among processes
    int colsPerProcess = vecCols / worldSize;                                                                      // Number of columns per process
    int extraCols = vecCols % worldSize;                                                                           // Number of extra columns to be distributed among processes
    int startCol = worldRank * colsPerProcess;                                                                     // Starting column index for the current process
    int endCol = (worldRank != worldSize - 1) ? startCol + colsPerProcess : startCol + colsPerProcess + extraCols; // Ending column index for the current process

    // Local computation
    int localSize = sparseMatrix.numRows * (endCol - startCol); // Number of elements in the local result vector
    std::vector<double> localResult(localSize, 0.0);
    // Iterate over the columns assigned to the current process
    for (int col = startCol; col < endCol; ++col)
    {
        // Iterate over the rows of the sparse matrix
        for (int i = 0; i < sparseMatrix.numRows; ++i)
        {
            // Iterate over the non-zero elements in the current row
            double sum = 0.0;
            for (int j = sparseMatrix.rowPtr[i]; j < sparseMatrix.rowPtr[i + 1]; ++j)
            {
                int sparseCol = sparseMatrix.colIndices[j];                // Column index of the non-zero element
                sum += sparseMatrix.values[j] * fatVector[sparseCol][col]; // Compute the result
            }
            localResult[i * (endCol - startCol) + (col - startCol)] = sum; // Store the result in the local result vector
        }
    }

    // =========================== FOR DEBUGGING ONLY - STOP LOCAL COMPUTATION TIMER ===============================
    // double computation_end = MPI_Wtime();
    // double local_computation_time = computation_end - computation_start;
    // =========================== FOR DEBUGGING ONLY - STOP LOCAL COMPUTATION TIMER ===============================

    // =========================== FOR DEBUGGING ONLY - START COMMUNICATION TIMER ==================================
    // Start timing for communication
    // double communication_start = MPI_Wtime();
    // =========================== FOR DEBUGGING ONLY - START COMMUNICATION TIMER ==================================

    // Preparation for Gather operation
    std::vector<int> recvCounts(worldSize), displacements(worldSize); // Number of elements to be received from each process, Displacement for each process
    if (worldRank == 0)
    {
        // Compute the number of elements to be received from each process
        int displacement = 0;
        for (int i = 0; i < worldSize; ++i)
        {
            int startColThisRank = i * colsPerProcess;                                                                                     // Starting column index for the current process
            int endColThisRank = (i != worldSize - 1) ? startColThisRank + colsPerProcess : startColThisRank + colsPerProcess + extraCols; // Ending column index for the current process
            recvCounts[i] = sparseMatrix.numRows * (endColThisRank - startColThisRank);                                                    // Number of elements to be received from the current process
            displacements[i] = displacement;                                                                                               // Displacement for the current process
            displacement += recvCounts[i];                                                                                                 // Update the displacement
        }
    }

    // Gather all local results into the root process
    std::vector<double> gatheredResults;
    if (worldRank == 0)
    {
        gatheredResults.resize(std::accumulate(recvCounts.begin(), recvCounts.end(), 0)); // Resize the vector to hold the final result
    }
    MPI_Gatherv(localResult.data(), localSize, MPI_DOUBLE,
                gatheredResults.data(), recvCounts.data(),
                displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD); // Gather the local results in the root process

    // =========================== FOR DEBUGGING ONLY - STOP COMMUNICATION TIMER ===================================
    // double communication_end = MPI_Wtime();
    // double local_communication_time = communication_end - communication_start;
    // =========================== FOR DEBUGGING ONLY - STOP COMMUNICATION TIMER ===================================

    // =========================== FOR DEBUGGING ONLY - COLLECTING AND ANALYSING PERFORMANCE DATA ==================
    // double total_computation_time = 0.0, total_communication_time = 0.0;
    // MPI_Reduce(&local_computation_time, &total_computation_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    // MPI_Reduce(&local_communication_time, &total_communication_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    // =========================== FOR DEBUGGING ONLY - COLLECTING AND ANALYSING PERFORMANCE DATA ==================

    // Reconstruct the final result matrix in the root process
    FatVector finalResult;
    if (worldRank == 0)
    {
        // =========================== FOR DEBUGGING ONLY - PRINTING PERFORMANCE DATA =================================
        // double avg_computation_time = total_computation_time / worldSize;
        // double avg_communication_time = total_communication_time / worldSize;
        // std::cout << "Column-wise Average Computation Time: " << avg_computation_time << std::endl;
        // std::cout << "Column-wise Average Communication Time: " << avg_communication_time << std::endl;
        // =========================== FOR DEBUGGING ONLY - PRINTING PERFORMANCE DATA =================================

        // Reconstruct the final result matrix
        finalResult.resize(sparseMatrix.numRows, std::vector<double>(vecCols, 0.0)); // Resize the final result matrix
        int resultIndex = 0;
        // Iterate over the processes
        for (int rank = 0; rank < worldSize; ++rank)
        {
            int numColsThisRank = (rank != worldSize - 1) ? colsPerProcess : colsPerProcess + extraCols; // Number of columns assigned to the current process
            int startColThisRank = rank * colsPerProcess;                                                // Starting column index for the current process

            // Iterate over the rows of the sparse matrix
            for (int row = 0; row < sparseMatrix.numRows; ++row)
            {
                // Iterate over the columns assigned to the current process
                for (int col = 0; col < numColsThisRank; ++col)
                {
                    finalResult[row][startColThisRank + col] = gatheredResults[resultIndex++]; // Reconstruct the final result matrix
                }
            }
        }
    }

    // Return the final result
    return (worldRank == 0) ? finalResult : FatVector{};
}