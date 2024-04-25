#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

python server.py &
sleep 1  # Sleep for 1s to allow the server to start

num_clients=5

for i in `seq 0 "$((num_clients - 1))"`; do
    echo "Starting client $i"
    python client.py --num-clients=${num_clients} --client-id=${i} &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait