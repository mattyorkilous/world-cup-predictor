from os import listdir
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import import_data as id
import clean_data as cd
import extract_features as ef
import select_model as sm
import simulate_tournament as st


def main() -> None:
    dir: str = [directory]

    file_names: list[str] = [
        x for x in listdir(dir + '/data') if x.endswith('.csv')
    ]

    match_records_wc, match_records_intl = id.read_datasets(dir, file_names)

    columns_to_keep: list[str] = [
        'date', 'tournament', 'team_home', 'goals_home', 
        'team_away', 'goals_away', 'home_stadium_or_not'
    ]

    match_records: pd.DataFrame = cd.clean_matches(
        match_records_intl,
        match_records_wc,
        columns_to_keep
    )

    elos_pre_match, elos = ef.get_pre_match_elos(match_records)

    matches: pd.DataFrame = ef.extract_match_features(
        match_records, 
        elos_pre_match
    )

    model: Pipeline = sm.select_model(matches)

    rng: np.random.Generator = np.random.default_rng(42)

    n_sims: int = 1000

    winners: list[str] = [
        st.simulate_tournament(
            model, 
            match_records, 
            elos,
            rng
        )[2]
        for i in range(n_sims)
    ]

    print(pd.Series(winners).value_counts())


if __name__ == '__main__':
    main()