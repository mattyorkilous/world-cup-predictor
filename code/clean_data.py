import pandas as pd
import numpy as np


def clean_matches(
        match_records_intl: pd.DataFrame, 
        match_records_wc: pd.DataFrame,
        columns_to_keep: list[str]
) -> pd.DataFrame:
    match_records_intl_clean: pd.DataFrame = clean_intl_matches(
        match_records_intl, 
        columns_to_keep
    )

    match_records_wc_clean: pd.DataFrame = clean_wc_matches(
        match_records_wc,
        columns_to_keep
    )

    match_records_combined: pd.DataFrame = pd.concat(
        [match_records_intl_clean, match_records_wc_clean]
    )

    match_records: pd.DataFrame = clean_match_records(match_records_combined)

    return match_records


def clean_intl_matches(
        match_records_intl: pd.DataFrame,
        columns_to_keep: list[str]
) -> pd.DataFrame:
    match_records_intl_clean: pd.DataFrame = (
        match_records_intl
            .assign(date=lambda df: pd.to_datetime(df['date']))
            .rename(
                columns={
                    'home_team': 'team_home',
                    'home_goals': 'goals_home',
                    'away_team': 'team_away',
                    'away_goals': 'goals_away' 
                }
            )
            [columns_to_keep]
    )

    return match_records_intl_clean


def clean_wc_matches(
        match_records_wc: pd.DataFrame,
        columns_to_keep: list[str]
) -> pd.DataFrame:
    match_records_wc_clean: pd.DataFrame = (
        match_records_wc
            .assign(
                tournament=lambda df: np.where(
                    df['stage'] == 'Group stage', 
                    'FIFA World Cup Group Stage',
                    'FIFA World Cup Knockout Stage'
                ),
                date=lambda df: pd.to_datetime(
                    pd.DataFrame({'year': df['year'], 'month': 1, 'day': 1})
                )
            )
            .rename(
                columns={
                    'host_team_or_not': 'home_stadium_or_not',
                    'home_team': 'team_home',
                    'home_goals': 'goals_home',
                    'away_team': 'team_away',
                    'away_goals': 'goals_away'
                }
            )
            [columns_to_keep]
    )

    return match_records_wc_clean


def clean_match_records(match_records: pd.DataFrame) -> pd.DataFrame:
    match_records = (
        match_records
            .sort_values('date')
            .reset_index(drop=True)
            .assign(
                match_id=lambda df: df.index,
                tournament=lambda df: classify_tournament(df['tournament']),
                result_home=lambda df: get_match_result(
                    df['goals_home'], 
                    df['goals_away']
                ),
                result_away=lambda df: get_match_result(
                    df['goals_away'], 
                    df['goals_home']
                ),
                goals_against_home=lambda df: df['goals_away'],
                goals_against_away=lambda df: df['goals_home'],
                opponent_home=lambda df: df['team_away'],
                opponent_away=lambda df: df['team_home'],
                **{
                    f'{result}_{team_type}': 
                    lambda df, result=result, team_type=team_type:
                        df[f'result_{team_type}'] == result
                    for result in ('win', 'loss') 
                    for team_type in ('home', 'away')
                },
                **{
                    col: lambda df, col=col: np.where(
                        df[col] >= 10,
                        10,
                        df[col]
                    ) 
                    for col in ('goals_home', 'goals_away')
                }
            )
            .rename(
                columns={
                    'goals_home': 'goals_for_home', 
                    'goals_away': 'goals_for_away'
                }
            )
    )

    return match_records


def classify_tournament(tournament: pd.Series) -> np.ndarray:
    tournament_classified: np.ndarray = np.select(
        (
            tournament.isin((
                'FIFA World Cup Knockout Stage', 
                'World Cup Knockouts'
            )),
            tournament.isin((
                'FIFA World Cup Group Stage',
                'World Cup Groups'
            )),
            tournament.isin((
                'CONMEBOL-UEFA Cup of Champions',
                'AFC Asian Cup',
                'African Cup of Nations',
                'Copa America',
                'Gold Cup',
                'UEFA Euro',
                'Confederation Cup'
            )),
            tournament.isin((
                'FIFA World Cup qualification',
                'AFC Asian Cup qualification',
                'African Cup of Nations qualification',
                'Gold Cup qualification',
                'UEFA Euro qualification',
                'Qualifiers'
            )),
            tournament.isin((
                'UEFA Nations League',
                'CONCACAF Nations League',
                'Nations League'
            ))
        ),
        (
            'World Cup Knockouts', 
            'World Cup Groups', 
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