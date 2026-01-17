import kagglehub
import polars as pl
from os import listdir
import numpy as np
from soccer.clean_data import clean_matches
from soccer.track_and_update_elos import track_and_update_elos
from soccer.extract_features import extract_match_features
from soccer.select_model import select_model
from soccer.simulate_draw import (
    get_qualified_teams, 
    simulate_group_draw, 
    get_group_stage_matchdays
)
from soccer.simulate_tournament import simulate_tournament


def main() -> None:
    # 1.Configuration
    rng: np.random.Generator = np.random.default_rng(42)
    
    n_sims: int = 10

    already_qualified: list[str] = [
        'United States', 'Mexico', 'Canada', 
        'Australia', 'Iran', 'Japan', 'Jordan', 'South Korea', 'Uzbekistan',
        'Argentina', 'Brazil', 'Ecuador',
        'New Zealand'
    ]

    already_drawn: dict[str, list[str]] = {
        'a': ['Mexico'],
        'b': ['Canada'],
        'd': ['United States']
    }
    
    team_confederations: pl.DataFrame = pl.read_csv(
        'lookup/confederation_mapping.csv'
    )

    path: str = kagglehub.dataset_download(
        'patateriedata/all-international-football-results'
    )

    # 2. Process data
    datasets_raw: dict[str, pl.DataFrame] = {
        file_name: pl.read_csv(f'{path}/{file_name}') 
        for file_name in listdir(path)
    }

    matches: pl.DataFrame = clean_matches(datasets_raw)

    matches_with_elos, elos = track_and_update_elos(matches)

    matches_with_features: pl.DataFrame = extract_match_features(
        matches_with_elos
    )

    # 3. Select model
    models, accuracy_summary = select_model(matches_with_features)

    print(accuracy_summary)

    # 4. Set up tournament
    teams_qualified: pl.DataFrame = get_qualified_teams(
        already_qualified,
        elos,
        team_confederations
    )

    groups: pl.DataFrame = simulate_group_draw(teams_qualified, already_drawn)

    matchdays_group_stage: list[pl.DataFrame] = get_group_stage_matchdays(
        groups,
        elos
    )

    # 5. Run simulations
    simulations = [
        simulate_tournament(matchdays_group_stage, models, matches, elos, rng)
        for _ in range(n_sims)
    ]

    winners: list[str] = [simulation[2] for simulation in simulations]

    print(pl.Series(winners).value_counts())
    

if __name__ == '__main__':
    main()