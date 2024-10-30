import pandas as pd
import numpy as np
from functools import reduce


def get_clean_matches(datasets_raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    country_names, matches_raw = datasets_raw.values()

    country_cols: list[str] = ['home_team', 'away_team', 'country']

    matches_all_names_replaced: pd.DataFrame = reduce(
        lambda df, col: replace_original_names(df, col, country_names),
        country_cols,
        matches_raw
    )

    matches_prepared: pd.DataFrame = (
        matches_all_names_replaced
            .reset_index()
            .rename(swap_underscore, axis=1)
    )

    matches = process_matches(matches_prepared)

    return matches


def replace_original_names(
        df: pd.DataFrame, 
        col: str, 
        country_names: pd.DataFrame
) -> pd.DataFrame:
    matches_names_replaced: pd.DataFrame = (
        df
            .assign(
                **{
                    col: lambda df: np.where(
                        df[col] == 'São Tomé and Príncipe', 
                        'Sao Tome and Principe', 
                        df[col]
                    )
                }
            )
            .merge(
                country_names, 
                how='left', 
                left_on=col, 
                right_on='original_name'
            )
            .assign(**{col: lambda df: df['current_name']})
            .drop(country_names.columns, axis=1)
    )
    
    return matches_names_replaced


def swap_underscore(text: str) -> str:
    if '_' in text:
        first, second = text.split('_', 1)

        return f"{second}_{first}"
    
    return text


def process_matches(matches_raw: pd.DataFrame) -> pd.DataFrame:
    matches: pd.DataFrame = (
        matches_raw
            .assign(
                tournament=lambda df: classify_tournament(df['tournament']),
                result_home=lambda df: get_match_result(
                    df['score_home'], 
                    df['score_away']
                ),
                result_away=lambda df: get_match_result(
                    df['score_away'], 
                    df['score_home']
                ),
                **{
                    col: lambda df, col=col: truncate_score(df, col)
                    for col in ('score_home', 'score_away')
                },
                goals_against_home=lambda df: df['score_away'],
                goals_against_away=lambda df: df['score_home'],
                opponent_home=lambda df: df['team_away'],
                opponent_away=lambda df: df['team_home'],
                **{
                    f'{result}_{team_type}': 
                    lambda df, result=result, team_type=team_type:
                        df[f'result_{team_type}'] == result
                    for result in ('win', 'loss') 
                    for team_type in ('home', 'away')
                }
            )
            .rename(
                columns={
                    'score_home': 'goals_for_home', 
                    'score_away': 'goals_for_away'
                }
            )
    )

    return matches


def classify_tournament(tournament: pd.Series) -> np.ndarray:
    tournament_classified: np.ndarray = np.select(
        (
            (tournament == 'World Cup'),
            tournament.isin((
                'Intercontinental Champ',
                'Artemio Franchi Trophy',
                'Asian Cup',
                'African Nations Cup',
                'Copa America',
                'Copa América',
                'CONCACAF Championship',
                'European Championship',
                'Confederations Cup',
                'Confederation Cup'
            )),
            tournament.isin((
                'World Cup qualifier'
                'Asian Cup qualifier',
                'African Nations Cup qualifier',
                'Copa América qualifier',
                'CONCACAF Champ qual',
                'European Championship qual'
                'Qualifiers'
            )),
            tournament.isin((
                'European Nations League',
                'European Nations League A',
                'European Nations League B',
                'European Nations League C',
                'European Nations League D',
                'CONCACAF Nations League',
                'CONCACAF Nations League A',
                'CONCACAF Nations League B',
                'CONCACAF Nations League C',
                'Nations League'
            ))
        ),
        (
            'World Cup', 
            'Confederation Cup',
            'Qualifiers',
            'Nations League'
        ),
        default='Friendly'
    )

    return tournament_classified


def get_match_result(
        goals_team: pd.Series, 
        goals_opponent: pd.Series
) -> np.ndarray:
    match_result: np.ndarray = np.select(
        (goals_team > goals_opponent, goals_team == goals_opponent),
        ('win', 'draw'),
        'loss'
    )

    return match_result


def truncate_score(df: pd.DataFrame, col: str) -> np.ndarray:
    score_truncated: np.ndarray = np.where(df[col] >= 10, 10, df[col])

    return score_truncated