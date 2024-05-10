# FedAvg without mask
python sim.py --name exp_1 --num_clients 3 \
    --num_rounds 2 --split_type balance_label \
    --config_file ./Config/stocks_hfl.yaml \
    --num_cpus 16 --num_gpus 0.9 --strategy fedavg

# FedAvg with mask
python sim.py --name exp_2 --num_clients 3 \
    --full_ratio 0.2 --num_rounds 2 \
    --split_type balance_label \
    --config_file ./Config/stocks_hfl_mask.yaml \
    --num_cpus 16 --num_gpus 0.9 --strategy fedavg

# FedWeightedAvg
python sim.py --name exp_3 --num_clients 3 \
    --full_ratio 0.2 --num_rounds 2 \
    --split_type balance_label \
    --config_file ./Config/stocks_hfl_mask.yaml \
    --num_cpus 16 --num_gpus 0.9 --strategy fedweightedavg

# FedNoAvg
python sim.py --name exp_4 --num_clients 3 \
    --full_ratio 0.2 --num_rounds 2 \
    --split_type balance_label \
    --config_file ./Config/stocks_hfl_mask.yaml \
    --num_cpus 16 --num_gpus 0.9 --strategy fednoavg

# FedHomoAvg
python sim.py --name exp_5 --num_clients 10 \
    --full_ratio 0.2 --repeat_prob 0.5 \
    --num_rounds 2 --split_type balance_label \
    --config_file ./Config/stocks_hfl_mask.yaml \
    --num_cpus 16 --num_gpus 0.9 --strategy fedhomoavg