#!/bin/bash

init_time=$(date +%s.%N)
interval=0.2 # interval between measurements in seconds
max_time=180 # maximum time for measuring memory usage in seconds

rm -f memory_usage.csv
echo "Time,Memory Usage (MB)" >> memory_usage.csv
echo "[SR7 INFO MEMORY] Start measuring memory usage"
while time_values<max_time; do
    # Get the current time
    time_values=$(echo "$(date +%s.%N) - $init_time" | bc)
    # Get the used memory in MB
    used_memory=$(free --mega | awk 'NR==2 {print $3}')
    # Store the used memory in a csv file
    echo "$time_values,$used_memory" >> memory_usage.csv
    sleep $interval
done
echo "[SR7 INFO MEMORY] Memory Usage (in MB) & Time (in seconds) \\\\"
