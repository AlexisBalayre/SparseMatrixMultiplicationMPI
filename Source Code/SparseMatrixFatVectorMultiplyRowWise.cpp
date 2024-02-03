#include <mpi.h>
#include "SparseMatrixFatVectorMultiplyRowWise.h"

/**
 * @brief Function to multiply a sparse matrix with a Fat Vector using row-wise distribution
 *
 * @param sparseMatrix  The sparse matrix to be multiplied
 * @param fatVector  The Fat Vector to be multiplied
 * @param vecCols  Number of columns in the Fat Vector
 * @return FatVector Result of the multiplication
 */
FatVector sparseMatrixFatVectorMultiplyRowWise(const SparseMatrix &sparseMatrix,
                                                   const FatVector &fatVector,
                                                   int vecCols)
{
    // Retrieve the rank and size of the MPI world
    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    // =========================== FOR DEBUGGING ONLY - START LOCAL COMPUTATION TIMER ===============================
    // double computation_start = MPI_Wtime();
    // =========================== FOR DEBUGGING ONLY - START LOCAL COMPUTATION TIMER ===============================

    // Distribute rows among processes
    int rowsCountPerProcess = sparseMatrix.numRows / worldSize;                      // Number of rows per process
    int extraRows = sparseMatrix.numRows % worldSize;                                // Number of extra rows to be distributed among processes
    int startRow = worldRank * rowsCountPerProcess + std::min(worldRank, extraRows); // Starting row index for the current process
    int endRow = startRow + rowsCountPerProcess + (worldRank < extraRows ? 1 : 0);   // Ending row index for the current process

    // Local computation
    int localSize = (endRow - startRow) * vecCols; // Number of elements in the local result vector
    std::vector<double> localResult(localSize);    // Local result vector

    // Iterate over the rows assigned to the current process
    for (int i = startRow; i < endRow; ++i)
    {
        // Iterate over the non-zero elements in the current row
        for (int j = sparseMatrix.rowPtr[i]; j < sparseMatrix.rowPtr[i + 1]; ++j)
        {
            int colIndex = sparseMatrix.colIndices[j]; // Column index of the non-zero element

            // Iterate over the columns of the Fat Vector
            for (int k = 0; k < vecCols; ++k)
            {
                int localIndex = (i - startRow) * vecCols + k;                                // Index of the element in the local result vector
                localResult[localIndex] += sparseMatrix.values[j] * fatVector[colIndex][k]; // Compute the result
            }
        }
    }

    // =========================== FOR DEBUGGING ONLY - STOP LOCAL COMPUTATION TIMER ===============================
    //  double computation_end = MPI_Wtime();
    //  double local_computation_time = computation_end - computation_start;
    // =========================== FOR DEBUGGING ONLY - STOP LOCAL COMPUTATION TIMER ===============================

    // =========================== FOR DEBUGGING ONLY - START COMMUNICATION TIMER ==================================
    //  Start timing for communication
    //  double communication_start = MPI_Wtime();
    // =========================== FOR DEBUGGING ONLY - START COMMUNICATION TIMER ==================================

    // Preparation for Gather operation
    std::vector<int> recvCounts(worldSize), displacements(worldSize);
    if (worldRank == 0)
    {
        int totalSize = 0; // Total number of elements to be received

        // Compute the number of elements to be received from each process
        for (int rank = 0; rank < worldSize; ++rank)
        {
            int startRowThisRank = rank * rowsCountPerProcess + std::min(rank, extraRows);            // Starting row index for the current process
            int endRowThisRank = startRowThisRank + rowsCountPerProcess + (rank < extraRows ? 1 : 0); // Ending row index for the current process
            recvCounts[rank] = (endRowThisRank - startRowThisRank) * vecCols;                         // Number of elements to be received from the current process
            displacements[rank] = totalSize;                                                          // Displacement for the current process
            totalSize += recvCounts[rank];                                                            // Update the total number of elements to be received
        }
    }

    // Gather all local results into the root process
    std::vector<double> gatheredResults;
    if (worldRank == 0)
    {
        gatheredResults.resize(recvCounts[0] * worldSize); // Resize the vector to hold all the results
    }
    MPI_Gatherv(localResult.data(), localSize, MPI_DOUBLE,
                gatheredResults.data(), recvCounts.data(),
                displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD); // Gather the local results in the root process

    // =========================== FOR DEBUGGING ONLY - STOP COMMUNICATION TIMER ===================================
    //  double communication_end = MPI_Wtime();
    //  double local_communication_time = communication_end - communication_start;
    // =========================== FOR DEBUGGING ONLY - STOP COMMUNICATION TIMER ===================================

    // =========================== FOR DEBUGGING ONLY - COLLECTING AND ANALYSING PERFORMANCE DATA ==================
    //  double total_computation_time = 0.0, total_communication_time = 0.0;
    //  MPI_Reduce(&local_computation_time, &total_computation_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    //  MPI_Reduce(&local_communication_time, &total_communication_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    // =========================== FOR DEBUGGING ONLY - COLLECTING AND ANALYSING PERFORMANCE DATA ==================

    // Reconstruct the final result matrix in the root process
    FatVector finalResult;
    if (worldRank == 0)
    {
        // =========================== FOR DEBUGGING ONLY - PRINTING PERFORMANCE DATA =================================
        //  double avg_computation_time = total_computation_time / worldSize;
        //  double avg_communication_time = total_communication_time / worldSize;
        //  std::cout << "Row-wise Average Computation Time: " << avg_computation_time << std::endl;
        //  std::cout << "Row-wise Average Communication Time: " << avg_communication_time << std::endl;
        // =========================== FOR DEBUGGING ONLY - PRINTING PERFORMANCE DATA =================================

        finalResult.resize(sparseMatrix.numRows, std::vector<double>(vecCols, 0.0)); // Resize the final result matrix

        // Iterate over the rows of the final result
        for (int i = 0, index = 0; i < sparseMatrix.numRows; ++i)
        {
            // Iterate over the columns of the final result
            for (int j = 0; j < vecCols; ++j, ++index)
            {
                finalResult[i][j] = gatheredResults[index]; // Copy the element of the final result
            }
        }
    }

    // Return the final result
    return (worldRank == 0) ? finalResult : FatVector{};
}