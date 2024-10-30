import pandas as pd
import numpy as np
from functools import reduce


def get_pre_match_elos(
        matches: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    match_records_list: list[pd.DataFrame] = [
        matches.iloc[[i]] for i in range(len(matches))
    ]

    elos: pd.DataFrame = get_initial_elos(matches)

    matches_so_far: pd.DataFrame = pd.DataFrame()

    elos_pre_match, elos = reduce(
        get_update_elos,
        match_records_list,
        (matches_so_far, elos)
    )

    return elos_pre_match, elos


def get_initial_elos(matches: pd.DataFrame) -> pd.DataFrame:
    elos_initial: pd.DataFrame = (
        pd.DataFrame()
            .assign(
                team=pd.concat([
                    matches['team_home'], 
                    matches['team_away']]
                ).unique(),
                elo=1000
            )
    )
    
    return elos_initial


def get_update_elos(
        matches_so_far__elos: tuple[pd.DataFrame, pd.DataFrame], 
        match_record: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches_so_far, elos = matches_so_far__elos

    match_record_joined: pd.DataFrame = join_pre_match_elos(match_record, elos)
    
    matches_so_far: pd.DataFrame = pd.concat([
        matches_so_far, 
        match_record_joined
    ])

    elos: pd.DataFrame = update_elos(elos, match_record_joined)

    return matches_so_far, elos


def join_pre_match_elos(
        match_records: pd.DataFrame, 
        elos: pd.DataFrame
) -> pd.DataFrame:
    match_records_joined: pd.DataFrame = (
        match_records
            .merge(
                elos.rename(columns={'team': 'team_home', 'elo': 'elo_home'}),
                on='team_home',
                how='left'
            )
            .merge(
                elos.rename(columns={'team': 'team_away', 'elo': 'elo_away'}),
                on='team_away',
                how='left'
            )
    )
    
    return match_records_joined


def update_elos(
        elos: pd.DataFrame, 
        match_record: pd.DataFrame
) -> pd.DataFrame:
    match_record_series: pd.Series = match_record.squeeze()

    elo_home, elo_away = get_updated_elos(match_record_series)

    elos_updated: pd.DataFrame = (
        elos
            .assign(
                elo=lambda df: np.select(
                    (
                        df['team'] == match_record_series['team_home'], 
                        df['team'] == match_record_series['team_away']
                    ), 
                    (elo_home, elo_away), 
                    default=df['elo']
                )
            )
    )

    return elos_updated


def get_updated_elos(match_record: pd.Series) -> tuple[float, float]:
    margin: int = abs(
        match_record['goals_for_home'] - match_record['goals_for_away']
    )

    margin_multiplier: float = (
        1 if margin == 0
            else 1.25 if margin == 1
            else 1.5 if margin == 2
            else 1.75 if margin == 3
            else 2
    )
    
    importance: int = (
        pd.Series(match_record['tournament'])
            .map(
                {
                    'World Cup': 55,
                    'Confederation Cup': 40,
                    'Qualifiers': 25,
                    'Nations League': 15,
                    'Friendly': 10
                }
            )
            [0]
    )

    elo_home: float = get_elo(
        match_record['elo_home'],
        match_record['elo_away'],
        importance,
        margin_multiplier,
        match_record['result_home']
    )

    elo_away: float = get_elo(
        match_record['elo_away'],
        match_record['elo_home'],
        importance,
        margin_multiplier,
        match_record['result_away']
    )

    return elo_home, elo_away


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
            + importance*margin_multiplier*(result_actual - result_expected)
    )
    
    return elo


def extract_match_features(
        matches: pd.DataFrame, 
        elos_pre_match: pd.DataFrame
) -> pd.DataFrame:
    team_matches: pd.DataFrame = melt_matches(matches)

    team_match_features: pd.DataFrame = extract_team_match_features(
        team_matches
    )

    match_features: pd.DataFrame = join_team_match_features(
        elos_pre_match, 
        team_match_features
    )

    return match_features


def melt_matches(
        matches: pd.DataFrame, 
        id_vars: list[str] = ['index', 'date']
) -> pd.DataFrame:
    value_vars: list[str] = list(
        matches.columns[
            matches.columns.str.contains('_home|_away')
        ]
    )

    matches_melted: pd.DataFrame = (
        matches
            .melt(id_vars=id_vars, value_vars=value_vars)
            .assign(
                var_name=lambda df: df['variable'].str.rsplit('_', n=1).str[0],
                team_type=lambda df: df['variable'].str.rsplit('_', n=1).str[1]
            )
            .pivot(
                index=(id_vars + ['team_type']), 
                columns='var_name', 
                values='value'
            )
            .reset_index()
            .rename_axis(None, axis=1)
    )

    return matches_melted


def extract_team_match_features(team_matches: pd.DataFrame) -> pd.DataFrame:
    team_match_features: pd.DataFrame = (
        team_matches
            .assign(
                **{
                    f'form_avg_{col}': lambda df, col=col: get_form_stat(
                        df, 
                        col
                    )
                    for col in ('win', 'loss', 'goals_for', 'goals_against')
                },
                **{
                    f'h2h_sum_{col}': lambda df, col=col: get_h2h_sum(df, col)
                    for col in ('win', 'loss')
                },
                count_cumulative=lambda df: df
                    .groupby(['team', 'opponent'])
                    .cumcount(),
                **{
                    f'h2h_rate_{result}': 
                        lambda df, result=result: get_h2h_rate(df, result)
                    for result in ('win', 'loss')
                },
                **{
                    col: lambda df, col=col: df[col].astype(int) 
                    for col in ('win', 'loss')
                }
            )
    )

    return team_match_features


def get_form_stat(df: pd.DataFrame, col: str) -> pd.Series:
    form_stat: pd.Series = (
        df
            .groupby('team')
            [col]
            .transform(
                lambda x: x
                    .shift(1)
                    .rolling(window=10, min_periods=1)
                    .mean()
                    .astype(float)
                    .fillna(0)
            )
    )

    return form_stat


def get_h2h_sum(df: pd.DataFrame, col: str) -> pd.Series:
    h2h_sum: pd.Series = (
        df
            .groupby(['team', 'opponent'])
            [col]
            .transform(
                lambda x: x
                    .shift(1)
                    .cumsum()
                    .astype(float)
                    .fillna(0)
            )
    )

    return h2h_sum


def get_h2h_rate(df: pd.DataFrame, result: str) -> pd.Series:
    h2h_rate: pd.Series = (
        (df[f'h2h_sum_{result}'] / df['count_cumulative'])
            .astype(float)
            .fillna(0)
    )

    return h2h_rate


def join_team_match_features(
        elos_pre_match: pd.DataFrame, 
        team_match_features: pd.DataFrame
) -> pd.DataFrame:
    values: list[str] = list(
        team_match_features.columns[
            team_match_features.columns.str.contains("form_avg_|h2h_rate_")
        ]
    )

    features: pd.DataFrame = (
        team_match_features
            .pivot(
                index='index', 
                columns='team_type', 
                values=values
            )
            .reset_index()
    )
    
    features.columns = flatten_columns(features.columns)

    features = features.rename_axis(None, axis=1)

    match_features = elos_pre_match.merge(
        features, 
        how='left', 
        on='index'
    )
    
    return match_features


def flatten_columns(df_columns: pd.Index) -> list[str]:
    columns_flattened: list[str] = [
        f'{i}_{j}' if j != '' else f'{i}' for i, j in df_columns
    ]

    return columns_flattened