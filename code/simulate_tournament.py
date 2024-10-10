import pandas as pd
import extract_features as ef
from sklearn.pipeline import Pipeline
import numpy as np
import clean_data as cd
from functools import reduce
from itertools import accumulate


def simulate_tournament(
        model: Pipeline, 
        match_records: pd.DataFrame, 
        elos: pd.DataFrame,
        rng: np.random.Generator
) -> tuple[pd.DataFrame, list[pd.DataFrame], str]:
    groups: pd.DataFrame = create_groups()

    matchdays_group_stage: list[pd.DataFrame] = get_group_stage_matchdays(
        groups
    )

    (
        group_table, 
        match_records_post_group, 
        elos_post_group
    ) = predict_group_stage(
        matchdays_group_stage, 
        model, 
        match_records, 
        elos,
        rng
    )

    bracket: pd.DataFrame = create_bracket(group_table)

    predictions_knockout_stage: list[pd.DataFrame] = predict_knockout_stage(
        bracket,
        model,
        elos_post_group,
        match_records_post_group,
        rng,
        predictions_so_far=[]
    )

    winner: str = get_winner(predictions_knockout_stage)

    return (group_table, predictions_knockout_stage, winner)


def create_groups() -> pd.DataFrame:
    groups: pd.DataFrame = (
        pd.DataFrame(
            {
                'group': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
                'teams': [
                    ['Qatar', 'Ecuador', 'Senegal', 'Netherlands'],
                    ['England', 'Iran', 'United States', 'Wales'],
                    ['Argentina', 'Saudi Arabia', 'Mexico', 'Poland'],
                    ['Denmark', 'Tunisia', 'France', 'Australia'],
                    ['Germany', 'Japan', 'Spain', 'Costa Rica'],
                    ['Morocco', 'Croatia', 'Belgium', 'Canada'],
                    ['Switzerland', 'Cameroon', 'Brazil', 'Serbia'],
                    ['Uruguay', 'South Korea', 'Portugal', 'Ghana']
                ]
            }
        )
    )

    return groups


def get_group_stage_matchdays(
        groups: pd.DataFrame, 
        host_nation: str = 'Qatar'
) -> list[pd.DataFrame]:
    matchup_patterns: list[list[tuple[int, ...]]] = get_matchup_patterns(
        groups
    )

    matchday_dates: pd.DatetimeIndex = pd.date_range('2022-11-20', periods=3)

    matchdays_group_stage: list[pd.DataFrame] = [
        get_group_stage_matchday(
            groups, 
            matchup_pattern, 
            matchday_date, 
            host_nation
        ) 
        for matchup_pattern, matchday_date in zip(
            matchup_patterns, 
            matchday_dates
        )
    ]

    return matchdays_group_stage


def get_matchup_patterns(groups: pd.DataFrame) -> list[list[tuple[int, ...]]]:
    n_teams: int = len(groups['teams'][0])

    matchup_patterns: list[list[tuple[int, ...]]] = [
        [(0, x), tuple(set(range(0, n_teams)) - {0, x})] 
        for x in range(1, n_teams)
    ]

    return matchup_patterns


def get_group_stage_matchday(
        groups: pd.DataFrame, 
        matchup_pattern: list[tuple[int, ...]],
        matchday_date: pd.Timestamp,
        host_nation: str
) -> pd.DataFrame:
    matchday: pd.DataFrame = (
        pd.concat(
            [
                get_group_matchday(
                    matchup_pattern, 
                    group, 
                    matchday_date, 
                    host_nation
                )
                for _, group in groups.iterrows()
            ]
        )
        .reset_index(drop=True)
        .assign(match_id=lambda df: df.index)
        [
            [
                'match_id', 'date', 'tournament', 'group', 
                'team_home', 'team_away', 'home_stadium_or_not'
            ]
        ]
    )

    return matchday


def get_group_matchday(
        matchup_pattern: list[tuple[int, ...]], 
        group: pd.Series,
        matchday_date: pd.Timestamp,
        host_nation: str
) -> pd.DataFrame:
    group_name, teams = group

    matchups: pd.DataFrame = (
        pd.DataFrame(
            [
                [teams[i] for i in matchup_pattern_element] 
                for matchup_pattern_element in matchup_pattern 
            ],
            columns=['team_home', 'team_away']
        )
        .assign(
            group=group_name,
            date=matchday_date,
            tournament='World Cup Groups',
            home_stadium_or_not=lambda df: (df['team_home'] == host_nation)
                .astype(int)
        )
    )

    return matchups


def predict_group_stage(
        matchdays_group_stage: list[pd.DataFrame], 
        model: Pipeline, 
        match_records: pd.DataFrame, 
        elos: pd.DataFrame,
        rng: np.random.Generator
) -> list[pd.DataFrame]:
    predictions_accumulated: list[list[pd.DataFrame]] = list(
        accumulate(
            matchdays_group_stage, 
            func=lambda x, y: predict_matchday_update_records(
                x, 
                y, 
                model=model,
                rng=rng
            ),
            initial=[pd.DataFrame(), match_records, elos]
        )
    )

    predictions_group_stage: list[pd.DataFrame] = (
        [x[0] for x in predictions_accumulated][1:]
    )

    group_table: pd.DataFrame = (
        ef.melt_matches(
            pd.concat(predictions_group_stage), 
            id_vars=['match_id', 'group', 'date']
        )
        .assign(
            points=lambda df: np.select(
                (df['win'] == True, df['loss'] == True),
                (3, 0),
                1
            ),
            gd=lambda df: df['goals_for'] - df['goals_against']
        )
        .groupby(['group', 'team'])
        [['points', 'gd', 'goals_for']]
        .agg('sum')
        .reset_index()
        .sort_values(
            ['group', 'points', 'gd', 'goals_for'], 
            ascending=[True, False, False, False]
        )
    )

    match_records_updated, elos_updated = [
        predictions_accumulated[-1][i] for i in (1, 2)
    ]

    return [group_table, match_records_updated, elos_updated]


def predict_matchday_update_records(
        preds__match_records__elos: list[pd.DataFrame],
        matchday: pd.DataFrame,
        model: Pipeline,
        rng: np.random.Generator
) -> list[pd.DataFrame]:
    _, match_records, elos = preds__match_records__elos

    matchday_features: pd.DataFrame = extract_matchday_features(
        matchday, 
        elos, 
        match_records
    )

    predictions_matchday = predict_matchday(
        matchday_features, 
        model,
        rng
    )

    predictions_matchday_list: list[pd.DataFrame] = [
        predictions_matchday.iloc[[i]] 
        for i in range(len(predictions_matchday))
    ]

    elos_updated = reduce(ef.update_elos, predictions_matchday_list, elos)

    match_records_updated = (
        pd.concat(
            [
                match_records, 
                predictions_matchday
                    .assign(
                        match_id=lambda df: range(
                            match_records['match_id'].max() + 1,
                            match_records['match_id'].max() + 1 + len(df)
                        )
                    )
                    [match_records.columns]
            ]
        )
    )

    return [predictions_matchday, match_records_updated, elos_updated]


def extract_matchday_features(
        matchday: pd.DataFrame, 
        elos: pd.DataFrame,
        match_records: pd.DataFrame
) -> pd.DataFrame:
    elos_matchday: pd.DataFrame = ef.join_pre_match_elos(matchday, elos)

    team_match_records: pd.DataFrame = ef.melt_matches(match_records)

    form_cols: list[str] = ['goals_for', 'goals_against', 'win', 'loss']

    team_form_matchday: pd.DataFrame = get_matchday_team_form(
        matchday,
        team_match_records, 
        form_cols
    )

    h2h_rate_cols: list[str] = ['win', 'loss']

    h2h_rates_matchday: pd.DataFrame = get_matchday_h2h_rates(
        matchday,
        team_match_records, 
        h2h_rate_cols
    )

    teams_matchday: pd.DataFrame = get_matchday_teams(
        matchday, 
        team_form_matchday, 
        form_cols
    )

    teams_matchday.columns = ef.flatten_columns(teams_matchday.columns)

    matchday_features: pd.DataFrame = join_matchday_features(
        elos_matchday,
        teams_matchday,
        h2h_rates_matchday
    )

    return matchday_features


def get_matchday_team_form(
        matchday: pd.DataFrame,
        team_match_records: pd.DataFrame,
        form_cols: list[str]
) -> pd.DataFrame:
    team_form_matchday: pd.DataFrame = (
        team_match_records
            .loc[
                lambda df: (df['team'].isin(matchday['team_home'])) 
                    | (df['team'].isin(matchday['team_away'])) 
            ]
            .groupby('team')
            .tail(10)
            .groupby('team')
            [form_cols]
            .agg(pd.Series.mean)
            .reset_index()
            .rename(
                columns={col: f'form_avg_{col}' for col in form_cols}
            )
    )

    return team_form_matchday


def get_matchday_h2h_rates(
        matchday: pd.DataFrame,
        team_match_records: pd.DataFrame, 
        h2h_rate_cols: list[str]
) -> pd.DataFrame:
    h2h_rates_matchday_1: pd.DataFrame = (
        team_match_records
            .merge(
                matchday[['team_home', 'team_away']], 
                how='left', 
                left_on=('team', 'opponent'), 
                right_on=('team_home', 'team_away'),
                indicator=True
            )
            .loc[lambda df: df['_merge'] == 'both']
            .groupby(['team', 'opponent'])
            [h2h_rate_cols]
            .agg(pd.Series.mean)
            .reset_index()
            .rename(
                columns={col: f'h2h_rate_{col}_home' for col in h2h_rate_cols}
            )
    )

    col_pairs: dict[str, str] = {
        'team': 'opponent',
        'opponent': 'team',
        'h2h_rate_win_home': 'h2h_rate_loss_home',
        'h2h_rate_loss_home': 'h2h_rate_win_home'
    }

    h2h_rates_matchday_2: pd.DataFrame = pd.DataFrame(
        {k: h2h_rates_matchday_1[v] for k, v in col_pairs.items()}
    )

    h2h_rates_matchday: pd.DataFrame = (
        pd.concat([h2h_rates_matchday_1, h2h_rates_matchday_2])
            .reset_index(drop=True)
    )

    return h2h_rates_matchday


def get_matchday_teams(
        matchday: pd.DataFrame,
        team_form: pd.DataFrame,
        form_cols: list[str]
) -> pd.DataFrame:
    teams_matchday: pd.DataFrame = (
        ef.melt_matches(matchday)
            .merge(team_form, how='left', on='team')
            .pivot(
                index='match_id', 
                columns='team_type', 
                values=[f'form_avg_{col}' for col in form_cols]
            )
            .reset_index()
    )

    return teams_matchday


def join_matchday_features(
        elos_matchday: pd.DataFrame,
        teams_matchday: pd.DataFrame,
        h2h_rates_matchday: pd.DataFrame
) -> pd.DataFrame:
    matchday_features: pd.DataFrame = (
        elos_matchday
            .merge(teams_matchday, how='left', on='match_id')
            .merge(
                h2h_rates_matchday, 
                how='left', 
                left_on=('team_home', 'team_away'), 
                right_on=('team', 'opponent')
            )
            .assign(
                **{
                    col: lambda df, col=col: df[col].fillna(0) 
                    for col in [
                        f'h2h_rate_{result}_home' 
                        for result in ('win', 'loss')
                    ]
                }
            )
    )

    return matchday_features


def predict_matchday(
        matchday_features: pd.DataFrame, 
        model: Pipeline,
        rng: np.random.Generator
) -> pd.DataFrame:
    scores_predicted: np.ndarray = rng.poisson(
        model.predict(matchday_features)
    )

    matchday_prediction: pd.DataFrame = cd.clean_match_records(
        matchday_features
            .assign(
                goals_home=[score[0] for score in scores_predicted],
                goals_away=[score[1] for score in scores_predicted]
            )
    )

    return matchday_prediction


def create_bracket(
        group_table: pd.DataFrame, 
        host_nation: str = 'Qatar'
) -> pd.DataFrame:
    bracket_template: pd.DataFrame = pd.DataFrame(
        {
            'match_id': [i // 2 for i in range(16)],
            'group': (list('abcdefghbadcfehg')), 
            'position': ([1, 2] * 8)
        }
    )

    teams_remaining: pd.DataFrame = (
        group_table
            .groupby('group')
            .head(2)
            .assign(position=lambda df: df.groupby('group').cumcount() + 1)
            [['group', 'position', 'team']]
    )

    bracket_teams: pd.DataFrame = (
        bracket_template
            .merge(teams_remaining, how='left', on=['group', 'position'])
    )

    bracket: pd.DataFrame = assign_team_types(bracket_teams, host_nation)

    return bracket


def assign_team_types(
        bracket_teams: pd.DataFrame,
        host_nation: str
) -> pd.DataFrame:
    bracket: pd.DataFrame = (
        bracket_teams
            .groupby('match_id')
            [['match_id', 'team']]
            .apply(lambda group: assign_team_type(group, host_nation))
            # Don't love this solution, since it requires that weird
            # column selection line to work around a warning. Look for 
            # a better solution with pivoting?
            .reset_index(drop=True)
    )

    return bracket


def assign_team_type(group: pd.DataFrame, host_nation: str) -> pd.DataFrame:
    if host_nation in group['team'].values:
        group_updated: pd.DataFrame = (
            group
                .assign(
                    team_type=lambda df: np.where(
                        df['team'] == host_nation, 
                        'home', 
                        'away'
                    )
                )
        )
    else:
        group_updated: pd.DataFrame = group.assign(team_type=['home', 'away'])
    
    return group_updated


def predict_knockout_stage(
        bracket: pd.DataFrame,
        model: Pipeline,
        elos: pd.DataFrame,
        match_records: pd.DataFrame,
        rng: np.random.Generator,
        predictions_so_far: list[pd.DataFrame],
        start_date: pd.Timestamp = pd.to_datetime('2022-11-23'),
        host_nation: str = 'Qatar',
) -> list[pd.DataFrame]:
    matchday: pd.DataFrame = get_knockout_stage_matchday(
        bracket, 
        host_nation,
        start_date
    )

    (
        predictions_matchday, 
        match_records_updated, 
        elos_updated
    ) = predict_matchday_update_records(
        [pd.DataFrame(), match_records, elos],
        matchday,
        model,
        rng
    )

    draws: pd.Series = (
        predictions_matchday
            .loc[
                lambda df: 
                    (df['win_home'] == False) & (df['loss_home'] == False)
            ]
            ['match_id']
    )

    samples: np.ndarray = rng.uniform(size=len(draws))

    shootout_wins_home = [
        draw for draw, sample in zip(draws, samples) if sample >= 0.5
    ]

    predictions_post_shootout = (
        predictions_matchday
            .assign(
                win_home=lambda df: np.where(
                    df['match_id'].isin(shootout_wins_home), 
                    True, 
                    df['win_home']
                ),
                win_away=lambda df: np.where(
                    (df['match_id'].isin(draws)) & 
                        (~df['match_id'].isin(shootout_wins_home)),
                    True,
                    df['win_away']
                )
            )
    )

    if (len(predictions_post_shootout) == 1):
        return (predictions_so_far + [predictions_post_shootout])

    bracket_teams_next: pd.DataFrame = (
        ef.melt_matches(predictions_post_shootout)
            .loc[lambda df: df['win']]
            .assign(
                match_id=[
                    i // 2 for i in range(len(predictions_post_shootout))
                ]
            )
    )

    bracket_next = assign_team_types(bracket_teams_next, host_nation)

    return predict_knockout_stage(
        bracket_next,
        model,
        elos_updated,
        match_records_updated,
        rng,
        predictions_so_far + [predictions_post_shootout],
        start_date + pd.Timedelta(days=1)
    )


def get_knockout_stage_matchday(
        bracket: pd.DataFrame, 
        host_nation: str,
        start_date: pd.Timestamp
) -> pd.DataFrame:
    matchday: pd.DataFrame = (
        bracket
            .assign(
                team_type=lambda df: ['team_' + x for x in df['team_type']]
            )
            [['match_id', 'team', 'team_type']]
            .pivot(columns='team_type', index='match_id', values='team')
            .reset_index()
            .rename_axis(None, axis=1)
            .assign(
                date=start_date,
                tournament='World Cup Knockouts',
                home_stadium_or_not=lambda df: (df['team_home'] == host_nation)
                .astype(int)
            )
            [
                [
                    'match_id', 'date', 'tournament', 'team_home', 
                    'team_away', 'home_stadium_or_not'
                ]
            ]
    )

    return matchday


def get_winner(predictions_knockout_stage: list[pd.DataFrame]) -> str:
    winner: str = (
        ef.melt_matches(predictions_knockout_stage[-1])
            .loc[lambda df: df['win']]
            ['team']
            .iloc[0]
    )

    return winner