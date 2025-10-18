import os
import glob
import pandas as pd
import yaml

def load_dataset(cfg_path: str, dataset: str = "olist", stage: str = "raw") -> dict:
    """
    Lê todos os CSVs de um dataset em um stage (raw, interim, processed).
    Retorna um dicionário {nome_arquivo: DataFrame}.
    """
    with open(cfg_path, "r") as f:
        data_cfg = yaml.safe_load(f)

    base_dir = data_cfg["datasets"][dataset][f"{stage}_dir"]
    csv_files = glob.glob(os.path.join(base_dir, "*.csv"))
    dataframes = {os.path.basename(f): pd.read_csv(f) for f in csv_files}
    return dataframes
