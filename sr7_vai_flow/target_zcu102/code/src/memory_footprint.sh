#!/bin/bash

init_time=$(date +%s.%N)
interval=0.2 # interval between measurements in seconds
max_time=180.0 # maximum time for measuring memory usage in seconds

rm -f memory_usage.csv
echo "Time,Memory Usage (MB)" >> memory_usage.csv
echo "[SR7 INFO MEMORY] Start measuring memory usage"

while (( $(echo "$init_time + $max_time > $(date +%s.%N)" | bc -l) )); do
    # Get the current time
    time_value=$(echo "$(date +%s.%N) - $init_time" | bc)
    # Get the used memory in MB
    used_memory=$(free --mega | awk 'NR==2 {print $3}')
    # Store the used memory in a csv file
    echo "$time_value,$used_memory" >> memory_usage.csv
    sleep $interval
done

echo "[SR7 INFO MEMORY] Memory Usage (in MB) & Time (in seconds)"
