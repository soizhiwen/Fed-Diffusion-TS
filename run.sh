###############################################################################
# Stock Dataset
###############################################################################

# FedAvg without mask
python sim.py --name exp_1 --num_clients 10 \
    --num_rounds 20 --split_type balance_label \
    --config_file ./Config/stocks_hfl.yaml \
    --num_cpus 16 --num_gpus 0.9 --strategy fedavg

# FedAvg with mask
python sim.py --name exp_2 --num_clients 10 \
    --full_ratio 0.2 --num_rounds 20 \
    --split_type balance_label \
    --config_file ./Config/stocks_hfl_mask.yaml \
    --num_cpus 16 --num_gpus 0.9 --strategy fedavg

# FedWeightedAvg
python sim.py --name exp_3 --num_clients 10 \
    --full_ratio 0.2 --num_rounds 20 \
    --split_type balance_label \
    --config_file ./Config/stocks_hfl_mask.yaml \
    --num_cpus 16 --num_gpus 0.9 --strategy fedweightedavg

# FedNoAvg
python sim.py --name exp_4 --num_clients 10 \
    --full_ratio 0.2 --num_rounds 20 \
    --split_type balance_label \
    --config_file ./Config/stocks_hfl_mask.yaml \
    --num_cpus 16 --num_gpus 0.9 --strategy fednoavg

# FedHomoAvg
python sim.py --name exp_5 --num_clients 10 \
    --full_ratio 0.2 --repeat_thold 0.5 \
    --num_rounds 20 --split_type balance_label \
    --config_file ./Config/stocks_hfl_mask.yaml \
    --num_cpus 16 --num_gpus 0.9 --strategy fedhomoavg


###############################################################################
# Energy Dataset
###############################################################################

# FedAvg without mask
python sim.py --name exp_6 --num_clients 10 \
    --num_rounds 20 --split_type balance_label \
    --config_file ./Config/energy_hfl.yaml \
    --num_cpus 16 --num_gpus 0.9 --strategy fedavg

# FedAvg with mask
python sim.py --name exp_7 --num_clients 10 \
    --full_ratio 0.2 --num_rounds 20 \
    --split_type balance_label \
    --config_file ./Config/energy_hfl_mask.yaml \
    --num_cpus 16 --num_gpus 0.9 --strategy fedavg

# FedWeightedAvg
python sim.py --name exp_8 --num_clients 10 \
    --full_ratio 0.2 --num_rounds 20 \
    --split_type balance_label \
    --config_file ./Config/energy_hfl_mask.yaml \
    --num_cpus 16 --num_gpus 0.9 --strategy fedweightedavg

# FedNoAvg
python sim.py --name exp_9 --num_clients 10 \
    --full_ratio 0.2 --num_rounds 20 \
    --split_type balance_label \
    --config_file ./Config/energy_hfl_mask.yaml \
    --num_cpus 16 --num_gpus 0.9 --strategy fednoavg

# FedHomoAvg
python sim.py --name exp_10 --num_clients 10 \
    --full_ratio 0.2 --repeat_thold 0.5 \
    --num_rounds 20 --split_type balance_label \
    --config_file ./Config/energy_hfl_mask.yaml \
    --num_cpus 16 --num_gpus 0.9 --strategy fedhomoavg