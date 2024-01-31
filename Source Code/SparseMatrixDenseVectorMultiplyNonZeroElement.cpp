#include "SparseMatrixDenseVectorMultiplyNonZeroElement.h"
#include <mpi.h>

/**
 * @brief Function to execute the sparse matrix-dense vector multiplication using non-zero element parallel algorithm
 *
 * @param sparseMatrix  Sparse matrix
 * @param denseVector  Dense vector
 * @param vecCols  Number of columns in the dense vector
 * @return DenseVector  Result of the multiplication
 */
DenseVector sparseMatrixDenseVectorMultiplyNonZeroElement(const SparseMatrix &sparseMatrix, const DenseVector &denseVector, int vecCols)
{
    // MPI Initialisation
    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    // =========================== FOR DEBUGGING ONLY - START LOCAL COMPUTATION TIMER ===============================
    // double computation_start = MPI_Wtime();
    // =========================== FOR DEBUGGING ONLY - START LOCAL COMPUTATION TIMER ===============================

    // Distribute non-zero elements among processes
    int totalNonZeroElements = sparseMatrix.values.size();
    int elementsPerProcess = totalNonZeroElements / worldSize;
    int extraElements = totalNonZeroElements % worldSize;
    int startIdx, endIdx;

    // Determine the starting and ending indices of the non-zero elements for the current process
    if (worldRank < extraElements)
    {
        startIdx = worldRank * (elementsPerProcess + 1);
        endIdx = startIdx + elementsPerProcess + 1;
    }
    else
    {
        startIdx = worldRank * elementsPerProcess + extraElements;
        endIdx = startIdx + elementsPerProcess;
    }

    // Precompute the row index mapping
    std::vector<int> rowIndexMap(sparseMatrix.values.size());
    for (int row = 0, idx = 0; row < sparseMatrix.rowPtr.size() - 1; ++row)
    {
        for (; idx < sparseMatrix.rowPtr[row + 1]; ++idx)
        {
            rowIndexMap[idx] = row;
        }
    }

    // Compute the multiplication for assigned non-zero elements
    std::vector<double> localResult(sparseMatrix.numRows * vecCols, 0.0);
    for (int idx = startIdx; idx < endIdx; ++idx)
    {
        int row = rowIndexMap[idx];
        int col = sparseMatrix.colIndices[idx];
        double value = sparseMatrix.values[idx];
        for (int k = 0; k < vecCols; ++k)
        {
            localResult[row * vecCols + k] += value * denseVector[col][k];
        }
    }

    // =========================== FOR DEBUGGING ONLY - STOP LOCAL COMPUTATION TIMER ===============================
    // double computation_end = MPI_Wtime();
    // double local_computation_time = computation_end - computation_start;
    // =========================== FOR DEBUGGING ONLY - STOP LOCAL COMPUTATION TIMER ===============================

    // =========================== FOR DEBUGGING ONLY - START COMMUNICATION TIMER ==================================
    // FOR DEBUGGING ONLY - START COMMUNICATION TIMER
    // double communication_start = MPI_Wtime();
    // =========================== FOR DEBUGGING ONLY - START COMMUNICATION TIMER ==================================

    // Initialize the final result only in the root process
    DenseVector finalResult;
    if (worldRank == 0)
    {
        finalResult.resize(sparseMatrix.numRows, std::vector<double>(vecCols, 0.0));
    }

    // Gather the local results in the root process
    std::vector<double> flatFinalResult(sparseMatrix.numRows * vecCols, 0.0);
    MPI_Reduce(localResult.data(), flatFinalResult.data(), sparseMatrix.numRows * vecCols, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // =========================== FOR DEBUGGING ONLY - STOP COMMUNICATION TIMER ===================================
    //  double communication_end = MPI_Wtime();
    //  double local_communication_time = communication_end - communication_start;
    // =========================== FOR DEBUGGING ONLY - STOP COMMUNICATION TIMER ===================================

    // =========================== FOR DEBUGGING ONLY - COLLECTING AND ANALYZING PERFORMANCE DATA ==================
    //  double total_computation_time = 0.0, total_communication_time = 0.0;
    //  MPI_Reduce(&local_computation_time, &total_computation_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    //  MPI_Reduce(&local_communication_time, &total_communication_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    // =========================== FOR DEBUGGING ONLY - COLLECTING AND ANALYZING PERFORMANCE DATA ==================

    // Reconstruct the finalResult from flatFinalResult in the root process
    if (worldRank == 0)
    {
        // =========================== FOR DEBUGGING ONLY - PRINTING PERFORMANCE DATA =================================
        //  double avg_computation_time = total_computation_time / worldSize;
        //  double avg_communication_time = total_communication_time / worldSize;
        //  std::cout << "Non-zero elements Average Computation Time: " << avg_computation_time << std::endl;
        //  std::cout << "Non-zero elements Average Communication Time: " << avg_communication_time << std::endl;
        // =========================== FOR DEBUGGING ONLY - PRINTING PERFORMANCE DATA =================================

        for (int i = 0; i < sparseMatrix.numRows; ++i)
        {
            std::copy(flatFinalResult.begin() + i * vecCols, flatFinalResult.begin() + (i + 1) * vecCols, finalResult[i].begin());
        }
    }

    return (worldRank == 0) ? finalResult : DenseVector{};
}
