###############################################################################
# Stock Dataset
###############################################################################

# FedAvg without mask
python sim.py --num_clients 10 --num_rounds 10 \
    --split_type balance_label --strategy fedavg \
    --config_file ./Config/stocks_hfl.yaml \
    --num_cpus 6 --num_gpus 0.3

# FedAvg with mask
python sim.py --num_clients --num_rounds 10 \
    --split_type balance_label --strategy fedavg \
    --config_file ./Config/stocks_hfl_mask.yaml \
    --full_ratio 0.2 --num_cpus 6 --num_gpus 0.3

# FedWeightedAvg
python sim.py --num_clients 10 --num_rounds 10 \
    --split_type balance_label --strategy fedweightedavg \
    --config_file ./Config/stocks_hfl_mask.yaml \
    --full_ratio 0.2 --num_cpus 6 --num_gpus 0.3

# FedNoAvg
python sim.py --num_clients 10 --num_rounds 10 \
    --split_type balance_label --strategy fednoavg \
    --config_file ./Config/stocks_hfl_mask.yaml \
    --full_ratio 0.2 --num_cpus 6 --num_gpus 0.3

# FedHomoAvg
python sim.py --num_clients 10 --num_rounds 10 \
    --split_type balance_label --strategy fedhomoavg \
    --config_file ./Config/stocks_hfl_mask.yaml \
    --full_ratio 0.2 --repeat_thold 0.5 \
    --num_cpus 6 --num_gpus 0.3

# FedDynaAvg
python sim.py --num_clients 10 --num_rounds 10 \
    --split_type balance_label --strategy feddynaavg \
    --config_file ./Config/stocks_hfl_mask.yaml \
    --full_ratio 0.2 --num_cpus 6 --num_gpus 0.3


###############################################################################
# Energy Dataset
###############################################################################

# FedAvg without mask
python sim.py --num_clients 10 --num_rounds 10 \
    --split_type balance_label --strategy fedavg \
    --config_file ./Config/energy_hfl.yaml \
    --num_cpus 6 --num_gpus 0.3

# FedAvg with mask
python sim.py --num_clients --num_rounds 10 \
    --split_type balance_label --strategy fedavg \
    --config_file ./Config/energy_hfl_mask.yaml \
    --full_ratio 0.2 --num_cpus 6 --num_gpus 0.3

# FedWeightedAvg
python sim.py --num_clients 10 --num_rounds 10 \
    --split_type balance_label --strategy fedweightedavg \
    --config_file ./Config/energy_hfl_mask.yaml \
    --full_ratio 0.2 --num_cpus 6 --num_gpus 0.3

# FedNoAvg
python sim.py --num_clients 10 --num_rounds 10 \
    --split_type balance_label --strategy fednoavg \
    --config_file ./Config/energy_hfl_mask.yaml \
    --full_ratio 0.2 --num_cpus 6 --num_gpus 0.3

# FedHomoAvg
python sim.py --num_clients 10 --num_rounds 10 \
    --split_type balance_label --strategy fedhomoavg \
    --config_file ./Config/energy_hfl_mask.yaml \
    --full_ratio 0.2 --repeat_thold 0.5 \
    --num_cpus 6 --num_gpus 0.3

# FedDynaAvg
python sim.py --num_clients 10 --num_rounds 10 \
    --split_type balance_label --strategy feddynaavg \
    --config_file ./Config/energy_hfl_mask.yaml \
    --full_ratio 0.2 --num_cpus 6 --num_gpus 0.3