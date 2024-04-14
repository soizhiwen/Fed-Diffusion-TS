# python main.py --name {name} --csv_file {data.csv} \
#     --config_file {config.yaml} --gpu 0 --train

python main.py --name exp_1 --csv_file ./Data/datasets/stock_data.csv \
    --config_file ./Config/stocks_latent.yaml --gpu 0 --train