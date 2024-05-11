###############################################################################
# Stock Dataset
###############################################################################

# FedAvg without mask
python sim.py --name exp --num_clients 10 \
    --num_rounds 10 --split_type balance_label \
    --config_file ./Config/stocks_hfl.yaml \
    --num_cpus 6 --num_gpus 0.3 --strategy fedavg

# FedAvg with mask
python sim.py --name exp --num_clients 10 \
    --full_ratio 0.2 --num_rounds 10 \
    --split_type balance_label \
    --config_file ./Config/stocks_hfl_mask.yaml \
    --num_cpus 6 --num_gpus 0.3 --strategy fedavg

# FedWeightedAvg
python sim.py --name exp --num_clients 10 \
    --full_ratio 0.2 --num_rounds 10 \
    --split_type balance_label \
    --config_file ./Config/stocks_hfl_mask.yaml \
    --num_cpus 6 --num_gpus 0.3 --strategy fedweightedavg

# FedNoAvg
python sim.py --name exp --num_clients 10 \
    --full_ratio 0.2 --num_rounds 10 \
    --split_type balance_label \
    --config_file ./Config/stocks_hfl_mask.yaml \
    --num_cpus 6 --num_gpus 0.3 --strategy fednoavg

# FedHomoAvg
python sim.py --name exp --num_clients 10 \
    --full_ratio 0.2 --repeat_thold 0.5 \
    --num_rounds 10 --split_type balance_label \
    --config_file ./Config/stocks_hfl_mask.yaml \
    --num_cpus 6 --num_gpus 0.3 --strategy fedhomoavg

# FedDynaAvg
python sim.py --name exp --num_clients 10 \
    --full_ratio 0.2 --num_rounds 10 \
    --split_type balance_label \
    --config_file ./Config/stocks_hfl_mask.yaml \
    --num_cpus 6 --num_gpus 0.3 --strategy feddynaavg


###############################################################################
# Energy Dataset
###############################################################################

# FedAvg without mask
python sim.py --name exp --num_clients 10 \
    --num_rounds 10 --split_type balance_label \
    --config_file ./Config/energy_hfl.yaml \
    --num_cpus 6 --num_gpus 0.3 --strategy fedavg

# FedAvg with mask
python sim.py --name exp --num_clients 10 \
    --full_ratio 0.2 --num_rounds 10 \
    --split_type balance_label \
    --config_file ./Config/energy_hfl_mask.yaml \
    --num_cpus 6 --num_gpus 0.3 --strategy fedavg

# FedWeightedAvg
python sim.py --name exp --num_clients 10 \
    --full_ratio 0.2 --num_rounds 10 \
    --split_type balance_label \
    --config_file ./Config/energy_hfl_mask.yaml \
    --num_cpus 6 --num_gpus 0.3 --strategy fedweightedavg

# FedNoAvg
python sim.py --name exp --num_clients 10 \
    --full_ratio 0.2 --num_rounds 10 \
    --split_type balance_label \
    --config_file ./Config/energy_hfl_mask.yaml \
    --num_cpus 6 --num_gpus 0.3 --strategy fednoavg

# FedHomoAvg
python sim.py --name exp --num_clients 10 \
    --full_ratio 0.2 --repeat_thold 0.5 \
    --num_rounds 10 --split_type balance_label \
    --config_file ./Config/energy_hfl_mask.yaml \
    --num_cpus 6 --num_gpus 0.3 --strategy fedhomoavg

# FedDynaAvg
python sim.py --name exp --num_clients 10 \
    --full_ratio 0.2 --num_rounds 10 \
    --split_type balance_label \
    --config_file ./Config/energy_hfl_mask.yaml \
    --num_cpus 6 --num_gpus 0.3 --strategy feddynaavg