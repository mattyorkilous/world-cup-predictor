import polars as pl
from typing import Any


def track_and_update_elos(
        matches: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    match_dicts: list[dict[str, Any]] = matches.to_dicts()

    match_dicts_with_elos: list[dict[str, Any]] = []

    elos_dict: dict[str, float] = get_initial_elos(matches)
    
    for match_dict in match_dicts:
        match_dicts_with_elos, elos_dict = get_and_update_elos(
            match_dict, match_dicts_with_elos, elos_dict
        )

    matches_with_elos: pl.DataFrame = pl.DataFrame(match_dicts_with_elos)

    elos: pl.DataFrame = pl.DataFrame({
        'team': elos_dict.keys(),
        'elo': elos_dict.values()
    })

    return matches_with_elos, elos


def get_initial_elos(matches: pl.DataFrame) -> dict[str, float]:
    elos_initial: dict[str, float] = {
        team: 1000.0 for team in (
            pl.concat([
                matches.get_column('team_home'),
                matches.get_column('team_away')
            ])
                .unique()
        )
    }
    
    return elos_initial


def get_and_update_elos(
        match_dict: dict[str, Any], 
        match_dicts_with_elos: list[dict[str, Any]], 
        elos_dict: dict[str, float]
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    match_dict_with_elos: dict[str, Any] = add_elos(match_dict, elos_dict)

    match_dicts_with_elos_updated: list[dict[str, Any]] = (
        match_dicts_with_elos + [match_dict_with_elos]
    )

    elos_dict_updated: dict[str, float] = update_elos(
        elos_dict, 
        match_dict_with_elos
    )

    return match_dicts_with_elos_updated, elos_dict_updated


def add_elos(
        match_dict: dict[str, Any],
        elos_dict: dict[str, float]
) -> dict[str, Any]:
    match_dict_with_elos: dict[str, Any] = match_dict.copy()

    match_dict_with_elos['elo_home'], match_dict_with_elos['elo_away'] = (
        elos_dict[match_dict['team_home']], 
        elos_dict[match_dict['team_away']]
    )

    return match_dict_with_elos


def update_elos(
        elos_dict: dict[str, float], 
        match_dict: dict[str, Any]
) -> dict[str, float]:
    elos_dict_updated: dict[str, float] = elos_dict.copy()

    elo_home: float = get_elo(
        match_dict['elo_home'],
        match_dict['elo_away'],
        match_dict['importance'],
        match_dict['margin_multiplier'],
        match_dict['result_home']
    )

    elo_away: float = get_elo(
        match_dict['elo_away'],
        match_dict['elo_home'],
        match_dict['importance'],
        match_dict['margin_multiplier'],
        match_dict['result_away']
    )

    elos_dict_updated[match_dict['team_home']] = elo_home

    elos_dict_updated[match_dict['team_away']] = elo_away

    return elos_dict_updated


def get_elo(
        elo_team: float,
        elo_opponent: float,
        importance: int,
        margin_multiplier: float,
        result: str
) -> float:
    result_actual: float = (
        1 if result == 'win'
        else 0.5 if result == 'draw'
        else 0
    )
    
    result_expected: float = (1 / (1 + 10**((elo_opponent-elo_team)/600)))

    elo: float = (
        elo_team 
        + int(importance)*margin_multiplier*(result_actual - result_expected)
    )
    
    return elo