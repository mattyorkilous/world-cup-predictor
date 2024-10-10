import pandas as pd


def read_datasets(dir: str, file_names: list[str]) -> list[pd.DataFrame]:
    datasets: list[pd.DataFrame] = [
        read_dataset(dir, file_name) for file_name in file_names
    ]

    return datasets


def read_dataset(path_project: str, file_name: str) -> pd.DataFrame:
    path: str = f'{path_project}/data/{file_name}'

    dataset: pd.DataFrame = (
        pd.read_csv(path)
            .rename(get_clean_col_name, axis=1)
    ) 
    
    return dataset


def get_clean_col_name(col: str) -> str:
    clean_col_name: str = col.lower().replace(' ', '_')

    return clean_col_name