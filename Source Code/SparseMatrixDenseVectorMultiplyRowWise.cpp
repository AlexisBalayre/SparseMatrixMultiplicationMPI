#include <mpi.h>
#include "SparseMatrixDenseVectorMultiplyRowWise.h"

/**
 * @brief Function to multiply a sparse matrix with a dense vector using row-wise distribution
 *
 * @param sparseMatrix  The sparse matrix to be multiplied
 * @param denseVector  The dense vector to be multiplied
 * @param vecCols  Number of columns in the dense vector
 * @return DenseVector Result of the multiplication
 */
DenseVector sparseMatrixDenseVectorMultiplyRowWise(const SparseMatrix &sparseMatrix,
                                                   const DenseVector &denseVector,
                                                   int vecCols)
{
    // MPI Initialisation
    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    // =========================== FOR DEBUGGING ONLY - START LOCAL COMPUTATION TIMER ===============================
    // double computation_start = MPI_Wtime();
    // =========================== FOR DEBUGGING ONLY - START LOCAL COMPUTATION TIMER ===============================

    int rowsCountPerProcess = sparseMatrix.numRows / worldSize;
    int extraRows = sparseMatrix.numRows % worldSize;

    int startRow = worldRank * rowsCountPerProcess + std::min(worldRank, extraRows);
    int endRow = startRow + rowsCountPerProcess + (worldRank < extraRows ? 1 : 0);

    // Local computation
    int localSize = (endRow - startRow) * vecCols;
    std::vector<double> localResult(localSize);

    for (int i = startRow; i < endRow; ++i)
    {
        for (int j = sparseMatrix.rowPtr[i]; j < sparseMatrix.rowPtr[i + 1]; ++j)
        {
            int colIndex = sparseMatrix.colIndices[j];
            for (int k = 0; k < vecCols; ++k)
            {
                int localIndex = (i - startRow) * vecCols + k;
                localResult[localIndex] += sparseMatrix.values[j] * denseVector[colIndex][k];
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

    // Preparation for MPI_Gatherv
    std::vector<int> recvCounts(worldSize), displacements(worldSize);
    if (worldRank == 0)
    {
        int totalSize = 0;
        for (int rank = 0; rank < worldSize; ++rank)
        {
            int startRowThisRank = rank * rowsCountPerProcess + std::min(rank, extraRows);
            int endRowThisRank = startRowThisRank + rowsCountPerProcess + (rank < extraRows ? 1 : 0);
            recvCounts[rank] = (endRowThisRank - startRowThisRank) * vecCols;
            displacements[rank] = totalSize;
            totalSize += recvCounts[rank];
        }
    }

    // Gather all local results into the root process
    std::vector<double> gatheredResults;
    if (worldRank == 0)
    {
        gatheredResults.resize(recvCounts[0] * worldSize);
    }
    MPI_Gatherv(localResult.data(), localSize, MPI_DOUBLE,
                gatheredResults.data(), recvCounts.data(),
                displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // =========================== FOR DEBUGGING ONLY - STOP COMMUNICATION TIMER ===================================
    //  double communication_end = MPI_Wtime();
    //  double local_communication_time = communication_end - communication_start;
    // =========================== FOR DEBUGGING ONLY - STOP COMMUNICATION TIMER ===================================

    // =========================== FOR DEBUGGING ONLY - COLLECTING AND ANALYZING PERFORMANCE DATA ==================
    //  double total_computation_time = 0.0, total_communication_time = 0.0;
    //  MPI_Reduce(&local_computation_time, &total_computation_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    //  MPI_Reduce(&local_communication_time, &total_communication_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    // =========================== FOR DEBUGGING ONLY - COLLECTING AND ANALYZING PERFORMANCE DATA ==================

    // Reconstruct the final result matrix in the root process
    DenseVector finalResult;
    if (worldRank == 0)
    {
        // =========================== FOR DEBUGGING ONLY - PRINTING PERFORMANCE DATA =================================
        //  double avg_computation_time = total_computation_time / worldSize;
        //  double avg_communication_time = total_communication_time / worldSize;
        //  std::cout << "Row-wise Average Computation Time: " << avg_computation_time << std::endl;
        //  std::cout << "Row-wise Average Communication Time: " << avg_communication_time << std::endl;
        // =========================== FOR DEBUGGING ONLY - PRINTING PERFORMANCE DATA =================================

        finalResult.resize(sparseMatrix.numRows, std::vector<double>(vecCols, 0.0));
        for (int i = 0, index = 0; i < sparseMatrix.numRows; ++i)
        {
            for (int j = 0; j < vecCols; ++j, ++index)
            {
                finalResult[i][j] = gatheredResults[index];
            }
        }
    }

    return (worldRank == 0) ? finalResult : DenseVector{};
}