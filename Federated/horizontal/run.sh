#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

num_clients=3
num_clusters=2
name="exp_1"
config="../../Config/stocks_hfl.yaml"

python server.py --num_clients $num_clients --num_clusters $num_clusters --config_file $config --multi_avg &
sleep 1  # Sleep for 1s to allow the server to start


for i in `seq 0 "$((num_clients - 1))"`; do
    echo "Starting client $i"
    python client.py --name $name --num_clients $num_clients --client_id $i --config_file $config &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait