import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sktime.datatypes import mtype

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from Utils.Data_utils.real_datasets import CustomDataset

ROOT_DIR = Path(os.path.abspath(__file__)).parent.parent
DATA_DIR = f"{ROOT_DIR}/Data/datasets"

if __name__ == "__main__":
    dataset = CustomDataset(
        name="stock",
        proportion=1.0,
        data_root=f"{DATA_DIR}/stock_data.csv",
        window=24,
        save2npy=True,
        neg_one_to_one=True,
        seed=123,
        period="train",
    )
    norm_data = np.load("./OUTPUT/samples/stock_norm_truth_24_train.npy")
    print(f"Before: {norm_data.shape}")

    df = pd.concat(
        [pd.DataFrame(x) for x in norm_data],
        keys=np.arange(norm_data.shape[0]),
    )

    x = np.zeros((norm_data.shape[0], norm_data.shape[-1]), dtype=pd.Series)
    for i in range(norm_data.shape[0]):
        for j in df.loc[i]:
            x[i][j] = df.loc[i][j]

    print(f"After: {x.shape}")

    np.save("./OUTPUT/samples/stock_norm_truth_24_nested_univ.npy", x)
    df = pd.DataFrame(x)
    print(mtype(df, as_scitype="Panel"))
