import kagglehub
import pandas as pd
from os import listdir
import numpy as np
from sklearn.pipeline import Pipeline
from clean_data import get_clean_matches
from extract_features import get_pre_match_elos, extract_match_features
from select_model import select_model
from simulate_tournament import simulate_tournament


def main() -> None:
    path: str = kagglehub.dataset_download(
        'patateriedata/all-international-football-results'
    )

    datasets_raw: dict[str, pd.DataFrame] = {
        file_name: pd.read_csv(f'{path}/{file_name}') 
        for file_name in listdir(path)
    }

    matches: pd.DataFrame = get_clean_matches(datasets_raw)

    matches = matches.loc[lambda df: df['date'] < '2022-11-20']

    elos_pre_match, elos = get_pre_match_elos(matches)

    match_features: pd.DataFrame = extract_match_features(
        matches, 
        elos_pre_match
    )

    model: Pipeline = select_model(match_features)

    rng: np.random.Generator = np.random.default_rng(42)

    n_sims: int = 1000

    winners: list[str] = [
        simulate_tournament(model, matches, elos, rng)[2]
        for i in range(n_sims)
    ]

    print(pd.Series(winners).value_counts())


if __name__ == '__main__':
    main()