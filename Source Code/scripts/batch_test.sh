#!/bin/bash

# Script to submit a batch of jobs to the cluster

# Path to the original script
original_script="mpi.sub"

# Maximum number of cores used for the job
max_cores=96

# Define a set of k values to test (Number of columns in the dense vector)
k_values=(1 3 6 9 12)

# Define a set of paths to test
paths=(
    "/mnt/beegfs/home/s425500/hpc/assignment/sparse-matrix/cop20k_A.mtx"
    "/mnt/beegfs/home/s425500/hpc/assignment/sparse-matrix/adder_dcop_32.mtx"
    "/mnt/beegfs/home/s425500/hpc/assignment/sparse-matrix/bcsstk17.mtx"
    "/mnt/beegfs/home/s425500/hpc/assignment/sparse-matrix/af23560.mtx"
    "/mnt/beegfs/home/s425500/hpc/assignment/sparse-matrix/amazon0302.mtx"
    "/mnt/beegfs/home/s425500/hpc/assignment/sparse-matrix/cavity10.mtx"
    "/mnt/beegfs/home/s425500/hpc/assignment/sparse-matrix/cage4.mtx"
    "/mnt/beegfs/home/s425500/hpc/assignment/sparse-matrix/dc1.mtx"
    "/mnt/beegfs/home/s425500/hpc/assignment/sparse-matrix/FEM_3D_thermal1.mtx"
    "/mnt/beegfs/home/s425500/hpc/assignment/sparse-matrix/mac_econ_fwd500.mtx"
    "/mnt/beegfs/home/s425500/hpc/assignment/sparse-matrix/mcfe.mtx"
    "/mnt/beegfs/home/s425500/hpc/assignment/sparse-matrix/mhd4800a.mtx"
    "/mnt/beegfs/home/s425500/hpc/assignment/sparse-matrix/olafu.mtx"
    "/mnt/beegfs/home/s425500/hpc/assignment/sparse-matrix/raefsky2.mtx"
    "/mnt/beegfs/home/s425500/hpc/assignment/sparse-matrix/rdist2.mtx"
    "/mnt/beegfs/home/s425500/hpc/assignment/sparse-matrix/thermal1.mtx"
    "/mnt/beegfs/home/s425500/hpc/assignment/sparse-matrix/thermomech_TK.mtx"
    "/mnt/beegfs/home/s425500/hpc/assignment/sparse-matrix/west2021.mtx"
    "/mnt/beegfs/home/s425500/hpc/assignment/sparse-matrix/lung2.mtx"
    "/mnt/beegfs/home/s425500/hpc/assignment/sparse-matrix/olm1000.mtx"
    "/mnt/beegfs/home/s425500/hpc/assignment/sparse-matrix/roadNet-PA.mtx"
)

# Loop over the k values
for k_value in "${k_values[@]}"; do
    # Loop over the paths of MTX files
    for path in "${paths[@]}"; do
        # Loop over the number of chunks
        for chunks in $(seq 1 $((max_cores / 16))); do
            # Loop over the number of cpus per chunk
            for cpus in $(seq 2 16); do
                # Calculate the total number of cores
                total_cores=$((chunks * cpus))
                # Check if the total number of cores is less than the maximum number of cores
                if [ $total_cores -le $max_cores ]; then
                    echo "Submitting job with $total_cores cores, $chunks chunks and $cpus cpus per chunk" 
                    echo "Path: $path"

                    # Create a unique job name
                    matrix_name=$(basename "$path") # Remove the path
                    sanitized_matrix_name=${matrix_name//[^a-zA-Z0-9_]/_} # Replace all non-alphanumeric characters with underscores
                    job_name="${sanitized_matrix_name}_k${k_value}_cores${total_cores}_chunks${chunks}_cpus${cpus}" # Add the k value to the job name

                    # Create a temporary submission script
                    temp_script="temp_${job_name}.sub"
                    cp "$original_script" "$temp_script"

                    # Replace the variables in the temporary script
                    sed -i "s|export k_value=.*|export k_value=${k_value}|" "$temp_script" # Export the k value
                    sed -i "s|export MATRIX_PATH=.*|export MATRIX_PATH=${path}|" "$temp_script" # Export the path to the MTX file
                    sed -i "s|#PBS -N .*|#PBS -N $job_name|" "$temp_script" # Set the job name
                    sed -i "s|#PBS -l select=.*|#PBS -l select=${chunks}:ncpus=$cpus:mpiprocs=$cpus|" "$temp_script" # Set the number of chunks and cpus per chunk

                    # Submit the job and get the job id
                    job_id=$(qsub "$temp_script")
                    echo "Job id: $job_id"

                    # Wait until the job is finished
                    while true; do
                        # Get the job status and duration
                        job_status=$(qstat -f "$job_id" | grep job_state | awk '{print $3}') # Get the job status
                        job_duration=$(qstat -f "$job_id" | grep resources_used.walltime | awk '{print $3}') # Get the job duration
                        job_duration_seconds=$(echo $job_duration | awk -F: '{ print ($1 * 3600) + ($2 * 60) + $3 }') # Convert the job duration to seconds
                        echo "Job status: $job_status"
                        echo "Job duration: $job_duration"

                        # If the job is finished, break the loop
                        if [ -z "$job_status" ]; then
                            break
                        fi

                        # if the job is running for more than 4 minutes, cancel it
                        if [ "$job_duration_seconds" -gt 240 ]; then
                            echo "Job is running for more than 4 minutes. Cancelling it."
                            qdel "$job_id"
                            break
                        fi

                        # Wait for 1 second
                        sleep 1
                    done

                    # Remove the temporary script
                    rm "$temp_script"
                fi
            done
        done
    done
done
