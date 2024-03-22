# High-Performance Computing Parallelization Strategies for Sparse Matrices

This project explores various parallelization strategies for multiplying sparse matrices by coarse vectors, using the Message Passing Interface (MPI) framework to achieve high performance on HPC systems. The focus is on comparing the efficiency, execution time and environmental impact of different methods, including sequential algorithms, row-based parallelism, column-based parallelism and non-zero element parallelism.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- An HPC system with MPI installed.
- Access to the PETSc library for some of the parallel computations.
- A compiler that supports MPI, such as mpicxx.

### Installation

1. Clone the repository to your local machine or HPC environment:

   ```bash
   git clone https://github.com/AlexisBalayre/SparseMatrixMultiplicationMPI
   ```

2. Navigate to the project directory:

   ```bash
   cd SparseMatrixMultiplicationMPI
   ```

3. Manual compilation:

   ```bash
   mpicxx -o main SparseMatrixFatVectorMultiply.cpp main.cpp utils.cpp -I${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include -L${PETSC_DIR}/${PETSC_ARCH}/lib -lpetsc
   ```

### Running the Tests

Execute the program with a specified number of processes and input matrix:

```bash
mpirun -np <number_of_processes> ./main <matrix_file_path> <k_value>
```

Replace `<number_of_processes>` with the desired number of MPI processes, `<matrix_file_path>` with the path to your sparse matrix file, and `<k_value>` with the number of columns in your fat vector.

## Project Structure

```graphql
Source Code /
   ├── scripts/                  # Shell scripts for automating tests and analyses
   │   ├── batch_test.sh         # Script for running batch tests
   │   ├── get_csv_all.sh        # Script to aggregate results into CSV format
   │   ├── get_csv_debug.sh      # Debug version of the CSV aggregation script
   │   ├── get_csv_specific.sh   # Script for extracting specific CSV data
   │   └── mpi.sub               # MPI submission script for job schedulers
   ├── MatrixDefinitions.h       # Header file defining the sparse matrix and fat vector
   ├── SparseMatrixFatVectorMultiply.h    # Interface for the sequential multiplication algorithm
   ├── SparseMatrixFatVectorMultiply.cpp  # Implementation of the sequential algorithm
   ├── SparseMatrixFatVectorMultiplyRowWise.h  # Interface for row-wise parallel multiplication
   ├── SparseMatrixFatVectorMultiplyRowWise.cpp # Implementation of row-wise parallel algorithm
   ├── SparseMatrixFatVectorMultiplyColumnWise.h # Interface for column-wise parallel multiplication
   ├── SparseMatrixFatVectorMultiplyColumnWise.cpp # Implementation of column-wise parallel algorithm
   ├── SparseMatrixFatVectorMultiplyNonZeroElement.h  # Interface for non-zero element parallel multiplication
   ├── SparseMatrixFatVectorMultiplyNonZeroElement.cpp # Implementation of non-zero element parallel algorithm
   ├── utils.h                   # Utility functions for matrix operations
   ├── utils.cpp                 # Implementation of utility functions
   └── main.cpp                  # Main program entry point
results/                  # Directory for storing output results
   ├── fat_vector_dim/       # Results categorized by fat vector dimensions
   │   └── <sparse_matrix><k><metric>.png  # Performance metrics visualization
   └── matrix_dim/           # Results categorized by sparse matrix dimensions
       └── <sparse_matrix><k><metric>.png  # Performance metrics visualization
```

## Built With

- [MPI](https://www.mpi-forum.org/) - The Message Passing Interface standard used for parallel computing.
- [PETSc](https://petsc.org/release/) - Portable, Extensible Toolkit for Scientific Computation.
