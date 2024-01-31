#include <mpi.h>
#include "SparseMatrixDenseVectorMultiplyColumnWise.h"
#include <numeric> // Pour std::accumulate

/**
 * @brief Function to execute the sparse matrix-dense vector multiplication using column-wise parallel algorithm
 *
 * @param sparseMatrix Sparse matrix
 * @param denseVector Dense vector
 * @param vecCols Number of columns in the dense vector
 * @return DenseVector Result of the multiplication
 */
DenseVector sparseMatrixDenseVectorMultiplyColumnWise(const SparseMatrix &sparseMatrix, const DenseVector &denseVector, int vecCols)
{
    // MPI Initialization
    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    /* // Start timing for computation
    double computation_start = MPI_Wtime(); */

    // Improved Workload Distribution
    int colsPerProcess = vecCols / worldSize;
    int extraCols = vecCols % worldSize;

    int startCol = worldRank * colsPerProcess;
    int endCol = (worldRank != worldSize - 1) ? startCol + colsPerProcess : startCol + colsPerProcess + extraCols;

    // Efficient Memory Allocation for Local Result
    int localSize = sparseMatrix.numRows * (endCol - startCol);
    std::vector<double> localResult(localSize, 0.0);

    // Optimized Local Computation
    for (int col = startCol; col < endCol; ++col)
    {
        for (int i = 0; i < sparseMatrix.numRows; ++i)
        {
            double sum = 0.0;
            for (int j = sparseMatrix.rowPtr[i]; j < sparseMatrix.rowPtr[i + 1]; ++j)
            {
                int sparseCol = sparseMatrix.colIndices[j];
                sum += sparseMatrix.values[j] * denseVector[sparseCol][col];
            }
            localResult[i * (endCol - startCol) + (col - startCol)] = sum;
        }
    }

    /*  // End timing for computation
     double computation_end = MPI_Wtime();
     double local_computation_time = computation_end - computation_start;

     // Start timing for communication
     double communication_start = MPI_Wtime(); */

    // Preparation for Gather operation
    std::vector<int> recvCounts(worldSize), displacements(worldSize);
    if (worldRank == 0)
    {
        int displacement = 0;
        for (int i = 0; i < worldSize; ++i)
        {
            int startColThisRank = i * colsPerProcess;
            int endColThisRank = (i != worldSize - 1) ? startColThisRank + colsPerProcess : startColThisRank + colsPerProcess + extraCols;
            recvCounts[i] = sparseMatrix.numRows * (endColThisRank - startColThisRank);
            displacements[i] = displacement;
            displacement += recvCounts[i];
        }
    }

    // Gather all local results into the root process
    std::vector<double> gatheredResults;
    if (worldRank == 0)
    {
        gatheredResults.resize(std::accumulate(recvCounts.begin(), recvCounts.end(), 0));
    }
    MPI_Gatherv(localResult.data(), localSize, MPI_DOUBLE,
                gatheredResults.data(), recvCounts.data(),
                displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* // End timing for communication
    double communication_end = MPI_Wtime();
    double local_communication_time = communication_end - communication_start;

    // Collecting and analyzing performance data
    double total_computation_time = 0.0, total_communication_time = 0.0;
    MPI_Reduce(&local_computation_time, &total_computation_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_communication_time, &total_communication_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); */

    // Reconstruct the final result matrix in the root process
    DenseVector finalResult;
    if (worldRank == 0)
    {
        /* double avg_computation_time = total_computation_time / worldSize;
        double avg_communication_time = total_communication_time / worldSize;
        std::cout << "Column-wise Average Computation Time: " << avg_computation_time << std::endl;
        std::cout << "Column-wise Average Communication Time: " << avg_communication_time << std::endl; */

        finalResult.resize(sparseMatrix.numRows, std::vector<double>(vecCols, 0.0));
        int resultIndex = 0;
        for (int rank = 0; rank < worldSize; ++rank)
        {
            int numColsThisRank = (rank != worldSize - 1) ? colsPerProcess : colsPerProcess + extraCols;
            int startColThisRank = rank * colsPerProcess;

            for (int row = 0; row < sparseMatrix.numRows; ++row)
            {
                for (int col = 0; col < numColsThisRank; ++col)
                {
                    finalResult[row][startColThisRank + col] = gatheredResults[resultIndex++];
                }
            }
        }
    }

    return (worldRank == 0) ? finalResult : DenseVector{};
}