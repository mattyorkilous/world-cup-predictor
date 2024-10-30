import pandas as pd
from extract_features import (
    join_pre_match_elos, melt_matches, update_elos, flatten_columns
)
from sklearn.pipeline import Pipeline
import numpy as np
from clean_data import process_matches
from functools import reduce
from itertools import accumulate


def simulate_tournament(
        model: Pipeline, 
        matches: pd.DataFrame, 
        elos: pd.DataFrame,
        rng: np.random.Generator
) -> tuple[pd.DataFrame, list[pd.DataFrame], str]:
    groups: pd.DataFrame = create_groups()

    matchdays_group_stage: list[pd.DataFrame] = get_group_stage_matchdays(
        groups
    )

    group_table, matches_post_group, elos_post_group = predict_group_stage(
        matchdays_group_stage, 
        model, 
        matches, 
        elos,
        rng
    )

    bracket: pd.DataFrame = create_bracket(group_table)

    predictions_knockout_stage: list[pd.DataFrame] = predict_knockout_stage(
        bracket,
        model,
        elos_post_group,
        matches_post_group,
        rng,
        predictions_so_far=[]
    )

    winner: str = get_winner(predictions_knockout_stage)

    return (group_table, predictions_knockout_stage, winner)


def create_groups() -> pd.DataFrame:
    groups: pd.DataFrame = (
        pd.DataFrame(
            {
                'group_name': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
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
        pd.concat([
            get_group_matchday(
                group,
                matchup_pattern,
                matchday_date, 
                host_nation
            )
            for _, group in groups.iterrows()
        ])
        .reset_index(drop=True)
        .reset_index()
        [[
            'index', 'date', 'tournament', 'group_name', 
            'team_home', 'team_away', 'country', 'neutral'
        ]]
    )

    return matchday


def get_group_matchday(
        group: pd.Series,
        matchup_pattern: list[tuple[int, ...]],
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
            group_name=group_name,
            date=matchday_date,
            tournament='World Cup',
            country=host_nation,
            neutral=lambda df: (df['team_home'] != host_nation)
        )
    )

    return matchups


def predict_group_stage(
        matchdays_group_stage: list[pd.DataFrame], 
        model: Pipeline, 
        matches: pd.DataFrame, 
        elos: pd.DataFrame,
        rng: np.random.Generator
) -> list[pd.DataFrame]:
    predictions_accumulated: list[list[pd.DataFrame]] = list(
        accumulate(
            matchdays_group_stage, 
            func=lambda predictions__matches__elos, matchday: 
                predict_matchday_update_records(
                    predictions__matches__elos, 
                    matchday, 
                    model=model,
                    rng=rng
                ),
            initial=[pd.DataFrame(), matches, elos]
        )
    )

    predictions_group_stage: list[pd.DataFrame] = (
        [x[0] for x in predictions_accumulated][1:]
    )

    group_table: pd.DataFrame = get_group_table(predictions_group_stage)

    matches_post_group, elos_post_group = [
        predictions_accumulated[-1][i] for i in (1, 2)
    ]

    return [group_table, matches_post_group, elos_post_group]


def predict_matchday_update_records(
        predictions__matches__elos: list[pd.DataFrame],
        matchday: pd.DataFrame,
        model: Pipeline,
        rng: np.random.Generator
) -> list[pd.DataFrame]:
    _, matches, elos = predictions__matches__elos

    matchday_features: pd.DataFrame = extract_matchday_features(
        matchday, 
        elos, 
        matches
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

    elos_updated = reduce(update_elos, predictions_matchday_list, elos)

    predictions_reindexed: pd.DataFrame = (
        predictions_matchday
            .assign(
                index=lambda df: range(
                    matches['index'].max() + 1,
                    matches['index'].max() + 1 + len(df)
                )
            )
            [matches.columns]
    )

    matches_updated = pd.concat([matches, predictions_reindexed])

    return [predictions_matchday, matches_updated, elos_updated]


def extract_matchday_features(
        matchday: pd.DataFrame, 
        elos: pd.DataFrame,
        matches: pd.DataFrame
) -> pd.DataFrame:
    elos_matchday: pd.DataFrame = join_pre_match_elos(matchday, elos)

    team_matches: pd.DataFrame = melt_matches(matches)

    form_cols: list[str] = ['goals_for', 'goals_against', 'win', 'loss']

    team_form_matchday: pd.DataFrame = get_matchday_team_form(
        matchday,
        team_matches, 
        form_cols
    )

    h2h_rate_cols: list[str] = ['win', 'loss']

    h2h_rates_matchday: pd.DataFrame = get_matchday_h2h_rates(
        matchday,
        team_matches, 
        h2h_rate_cols
    )

    teams_matchday: pd.DataFrame = get_matchday_teams(
        matchday, 
        team_form_matchday, 
        form_cols
    )

    matchday_features: pd.DataFrame = join_matchday_features(
        elos_matchday,
        teams_matchday,
        h2h_rates_matchday
    )

    return matchday_features


def get_matchday_team_form(
        matchday: pd.DataFrame,
        team_matches: pd.DataFrame,
        form_cols: list[str]
) -> pd.DataFrame:
    team_form_matchday: pd.DataFrame = (
        team_matches
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
        team_matches: pd.DataFrame, 
        h2h_rate_cols: list[str]
) -> pd.DataFrame:
    h2h_rates_matchday_1: pd.DataFrame = (
        team_matches
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
        melt_matches(matchday)
            .merge(team_form, how='left', on='team')
            .pivot(
                index='index', 
                columns='team_type', 
                values=[f'form_avg_{col}' for col in form_cols]
            )
            .reset_index()
    )

    teams_matchday.columns = flatten_columns(teams_matchday.columns)

    return teams_matchday


def join_matchday_features(
        elos_matchday: pd.DataFrame,
        teams_matchday: pd.DataFrame,
        h2h_rates_matchday: pd.DataFrame
) -> pd.DataFrame:
    matchday_features: pd.DataFrame = (
        elos_matchday
            .merge(teams_matchday, how='left', on='index')
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

    matchday_scores: pd.DataFrame = (
        matchday_features
            .assign(
                score_home=[score[0] for score in scores_predicted],
                score_away=[score[1] for score in scores_predicted]
            )
    )

    matchday_prediction: pd.DataFrame = process_matches(matchday_scores)

    return matchday_prediction


def get_group_table(
        predictions_group_stage: list[pd.DataFrame]
) -> pd.DataFrame:
    group_table: pd.DataFrame = (
        melt_matches(
            pd.concat(predictions_group_stage), 
            id_vars=['index', 'group_name', 'date']
        )
        .assign(
            points=lambda df: np.select(
                (df['win'] == True, df['loss'] == True),
                (3, 0),
                1
            ),
            gd=lambda df: df['goals_for'] - df['goals_against']
        )
        .groupby(['group_name', 'team'])
        [['points', 'gd', 'goals_for']]
        .agg('sum')
        .reset_index()
        .sort_values(
            ['group_name', 'points', 'gd', 'goals_for'], 
            ascending=[True, False, False, False]
        )
    )

    return group_table


def create_bracket(
        group_table: pd.DataFrame, 
        host_nation: str = 'Qatar'
) -> pd.DataFrame:
    bracket_template: pd.DataFrame = pd.DataFrame(
        {
            'index': [i // 2 for i in range(16)],
            'group_name': (list('abcdefghbadcfehg')), 
            'position': ([1, 2] * 8)
        }
    )

    teams_remaining: pd.DataFrame = (
        group_table
            .groupby('group_name')
            .head(2)
            .assign(
                position=lambda df: df.groupby('group_name').cumcount() + 1
            )
            [['group_name', 'position', 'team']]
    )

    bracket_teams: pd.DataFrame = (
        bracket_template
            .merge(teams_remaining, how='left', on=['group_name', 'position'])
    )

    bracket: pd.DataFrame = assign_team_types(bracket_teams, host_nation)

    return bracket


def assign_team_types(
        bracket_teams: pd.DataFrame,
        host_nation: str
) -> pd.DataFrame:
    bracket: pd.DataFrame = (
        bracket_teams
            .groupby('index')
            [['index', 'team']]
            .apply(
                lambda data_group: assign_team_type(data_group, host_nation)
            )
            .reset_index(drop=True)
    )

    return bracket


def assign_team_type(
        data_group: pd.DataFrame, 
        host_nation: str
) -> pd.DataFrame:
    if host_nation in data_group['team'].values:
        data_group_updated: pd.DataFrame = (
            data_group
                .assign(
                    team_type=lambda df: np.where(
                        df['team'] == host_nation, 
                        'home', 
                        'away'
                    )
                )
        )
    else:
        data_group_updated: pd.DataFrame = (
            data_group
                .assign(team_type=['home', 'away'])
        )
    
    return data_group_updated


def predict_knockout_stage(
        bracket: pd.DataFrame,
        model: Pipeline,
        elos: pd.DataFrame,
        matches: pd.DataFrame,
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
        matches_updated, 
        elos_updated
    ) = predict_matchday_update_records(
        [pd.DataFrame(), matches, elos],
        matchday,
        model,
        rng
    )

    draws: pd.Series = locate_draws(predictions_matchday)

    predictions_post_shootout: pd.DataFrame = predict_shootouts(
        predictions_matchday, 
        draws, 
        rng
    )

    if (len(predictions_post_shootout) == 1):
        return (predictions_so_far + [predictions_post_shootout])

    bracket_teams_next_round: pd.DataFrame = get_next_round(
        predictions_post_shootout
    )

    bracket_next_round = assign_team_types(
        bracket_teams_next_round, 
        host_nation
    )

    return predict_knockout_stage(
        bracket_next_round,
        model,
        elos_updated,
        matches_updated,
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
            [['index', 'team', 'team_type']]
            .pivot(columns='team_type', index='index', values='team')
            .reset_index(drop=True)
            .rename_axis(None, axis=1)
            .assign(
                date=start_date,
                tournament='World Cup',
                country = host_nation,
                neutral=lambda df: (df['team_home'] != host_nation)
            )
            .reset_index()
            [[
                'index', 'date', 'tournament', 'team_home', 
                'team_away', 'country', 'neutral'
            ]]
    )

    return matchday


def locate_draws(predictions_matchday: pd.DataFrame) -> pd.Series:
    draws: pd.Series = (
        predictions_matchday
            .loc[
                lambda df: (
                    df['win_home'] == False) & (df['loss_home'] == False
                )
            ]
            ['index']
    )

    return draws


def predict_shootouts(
        predictions_matchday: pd.DataFrame, 
        draws: pd.Series, 
        rng: np.random.Generator
) -> pd.DataFrame:
    samples: np.ndarray = rng.uniform(size=len(draws))

    shootout_wins_home = [
        draw for draw, sample in zip(draws, samples) if sample >= 0.5
    ]

    predictions_post_shootout: pd.DataFrame = (
        predictions_matchday
            .assign(
                win_home=lambda df: np.where(
                    df['index'].isin(shootout_wins_home), 
                    True, 
                    df['win_home']
                ),
                win_away=lambda df: np.where(
                    (df['index'].isin(draws)) & 
                        (~df['index'].isin(shootout_wins_home)),
                    True,
                    df['win_away']
                )
            )
    )

    return predictions_post_shootout


def get_next_round(predictions_post_shootout: pd.DataFrame) -> pd.DataFrame:
    bracket_teams_next_round: pd.DataFrame = (
        melt_matches(predictions_post_shootout)
            .loc[lambda df: df['win']]
            .assign(
                index=[i // 2 for i in range(len(predictions_post_shootout))]
            )
    )

    return bracket_teams_next_round


def get_winner(predictions_knockout_stage: list[pd.DataFrame]) -> str:
    winner: str = (
        melt_matches(predictions_knockout_stage[-1])
            .loc[lambda df: df['win']]
            ['team']
            .iloc[0]
    )

    return winner