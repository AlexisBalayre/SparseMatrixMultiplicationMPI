#!/bin/bash

# Name of the CSV file to write the data to
output_csv="results_debug.csv"

# Headers for the CSV file
echo "File Name,Cores Number,Sparse Matrix,Fat Vector,Serial Algo Execution time,Setup time,Row-wise Average Communication Time,Row-wise Average Computation Time,Row-wise Execution time,Row-wise Result,Column-wise Average Communication Time,Column-wise Average Computation Time,Column-wise Execution time,Column-wise Result,Non-zero elements Average Communication Time,Non-zero elements Average Computation Time,Non-zero Elements Execution time,Non-zero Elements Result,PETSc Setup time,PETSc Execution time,PETSc Conversion time,PETSc Result" >$output_csv

# Loop over the output files (Debug Files Only)
for file in debug*.o*; do
    # Check that the file is valid and that it is a result file
    if [[ -s $file && $file == *mtx* ]]; then
        # Extract the job name and the number of cores from the file name
        job_name=$(basename "$file" | sed -e 's/\.[^.]*$//') # Remove file extension
        num_cores=$(echo $file | grep -oP '(?<=_cores)\d+')  # Extract the number of cores from the file name

        # Extract the matrix size and the vector size from the file
        matrix_size=$(grep "Matrix size" $file | awk '{print $3}' | sed 's/size://')
        vector_size=$(grep "Vector size" $file | awk '{print $3}' | sed 's/size://')

        # Extract the serial execution time from the file
        serial_time=$(grep "Serial Algo Execution time" $file | awk '{print $5}')

        # Broadcast time of Sparse Matrix and Fat Vector
        setup_time=$(grep "Broadcast time" $file | awk '{print $3}')

        # Row-wise Data
        row_wise_communication_time=$(grep "Row-wise Average Communication Time" $file | awk '{print $5}') # Extract the row-wise average communication time from the file
        row_wise_computation_time=$(grep "Row-wise Average Computation Time" $file | awk '{print $5}')     # Extract the row-wise average computation time from the file
        row_wise_execution_time=$(grep "Row-wise Execution time" $file | awk '{print $4}')                 # Extract the row-wise execution time from the file
        row_wise_result=$(grep "Row-wise: Results are" $file | awk '{print $5}')                           # Extract the row-wise result from the file
        row_wise_result=$(if [ $row_wise_result == "same!" ]; then echo "same"; else echo "different"; fi) # Convert the row-wise result to a boolean

        # Column-wise Data
        col_wise_communication_time=$(grep "Column-wise Average Communication Time" $file | awk '{print $5}') # Extract the column-wise average communication time from the file
        col_wise_computation_time=$(grep "Column-wise Average Computation Time" $file | awk '{print $5}')     # Extract the column-wise average computation time from the file
        col_wise_execution_time=$(grep "Column-wise Execution time" $file | awk '{print $4}')                 # Extract the column-wise execution time from the file
        col_wise_result=$(grep "Column-wise: Results are" $file | awk '{print $5}')                           # Extract the column-wise result from the file
        col_wise_result=$(if [ $col_wise_result == "same!" ]; then echo "same"; else echo "different"; fi)    # Convert the column-wise result to a boolean

        # Non-zero element Data
        nonzero_communication_time=$(grep "Non-zero elements Average Communication Time" $file | awk '{print $6}') # Extract the non-zero elements average communication time from the file
        nonzero_computation_time=$(grep "Non-zero elements Average Computation Time" $file | awk '{print $6}')     # Extract the non-zero elements average computation time from the file
        nonzero_execution_time=$(grep "Non-zero Elements Execution time" $file | awk '{print $5}')                 # Extract the non-zero elements execution time from the file
        nonzero_result=$(grep "Non-zero Elements: Results are" $file | awk '{print $6}')                           # Extract the non-zero elements result from the file
        nonzero_result=$(if [ $nonzero_result == "same!" ]; then echo "same"; else echo "different"; fi)           # Convert the non-zero elements result to a boolean

        # PETSc Data
        petsc_setup=$(grep "PETSc Setup time" $file | awk '{print $4}')                              # Extract the PETSc setup time from the file
        petsc_execution_time=$(grep "PETSc Execution time" $file | awk '{print $4}')                 # Extract the PETSc execution time from the file
        petsc_conversion_time=$(grep "PETSc Conversion time" $file | awk '{print $4}')               # Extract the PETSc conversion time from the file
        petsc_result=$(grep "PETSc: Results are" $file | awk '{print $5}')                           # Extract the PETSc result from the file
        petsc_result=$(if [ $petsc_result == "same!" ]; then echo "same"; else echo "different"; fi) # Convert the PETSc result to a boolean

        # Write the extracted data to the CSV file
        echo "$job_name,$num_cores,$matrix_size,$vector_size,$serial_time,$setup_time,$row_wise_communication_time,$row_wise_computation_time,$row_wise_execution_time,$row_wise_result,$col_wise_communication_time,$col_wise_computation_time,$col_wise_execution_time,$col_wise_result,$nonzero_communication_time,$nonzero_computation_time,$nonzero_execution_time,$nonzero_result,$petsc_setup,$petsc_execution_time,$petsc_conversion_time,$petsc_result" >>$output_csv
    fi
done

echo "The data was successfully written in $output_csv"
