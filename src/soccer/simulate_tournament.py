import polars as pl
import numpy as np
from sklearn.pipeline import Pipeline
from typing import Any
from soccer.track_and_update_elos import update_elos
from soccer.extract_features import unpivot_matches
from soccer.clean_data import process_match_scores
from soccer.simulate_draw import get_matchday


def simulate_tournament(
        matchdays_group_stage: list[pl.DataFrame],
        models: dict[str, Pipeline], 
        matches: pl.DataFrame, 
        elos: pl.DataFrame,
        rng: np.random.Generator
) -> tuple[pl.DataFrame, list[pl.DataFrame], str]:
    group_table, matches_post_group, elos_post_group = predict_group_stage(
        matchdays_group_stage, 
        models, 
        matches, 
        elos,
        rng
    )

    bracket: pl.DataFrame = create_bracket(group_table)

    predictions_knockout_stage: list[pl.DataFrame] = predict_knockout_stage(
        bracket,
        models,
        elos_post_group,
        matches_post_group,
        rng
    )

    winner: str = get_winner(predictions_knockout_stage)

    return (group_table, predictions_knockout_stage, winner)


def predict_group_stage(
        matchdays_group_stage: list[pl.DataFrame], 
        models: dict[str, Pipeline], 
        matches: pl.DataFrame, 
        elos: pl.DataFrame,
        rng: np.random.Generator
) -> tuple[pl.DataFrame, ...]:
    predictions_group_stage: list[pl.DataFrame] = []

    for matchday in matchdays_group_stage:
        predictions_matchday, matches, elos = predict_matchday_update_records(
            matchday,
            matches,
            elos,
            models,
            rng
        )

        predictions_group_stage = (
            predictions_group_stage + [predictions_matchday]
        )

    group_table: pl.DataFrame = get_group_table(predictions_group_stage)

    return group_table, matches, elos


def predict_matchday_update_records(
        matchday: pl.DataFrame,
        matches: pl.DataFrame,
        elos: pl.DataFrame,
        models: dict[str, Pipeline],
        rng: np.random.Generator
) -> tuple[pl.DataFrame, ...]:
    matchday_features: pl.DataFrame = extract_matchday_features(
        matchday, 
        matches
    )

    predictions_matchday: pl.DataFrame = predict_matchday(
        matchday_features, 
        models,
        rng
    )

    match_dicts: list[dict[str, Any]] = predictions_matchday.to_dicts()

    elos_dict: dict[str, float] = dict(elos.iter_rows())

    for match_dict in match_dicts:
        elos_dict = update_elos(elos_dict, match_dict)

    elos_updated: pl.DataFrame = pl.DataFrame({
        'team': elos_dict.keys(),
        'elo': elos_dict.values()
    })

    id_start: int = matches.get_column('match_id').max() + 1 # type: ignore

    predictions_reindexed: pl.DataFrame = (
        predictions_matchday
            .drop('match_id')
            .with_row_index(
                'match_id', 
                offset=id_start
            )
            .select(matches.columns)
    )

    matches_updated: pl.DataFrame = pl.concat([matches, predictions_reindexed])

    return predictions_matchday, matches_updated, elos_updated


def extract_matchday_features(
        matchday: pl.DataFrame,
        matches: pl.DataFrame
) -> pl.DataFrame:
    index: list[str] = [
        col for col in matchday.columns 
        if '_home' not in col and '_away' not in col
    ]

    matchday_by_team: pl.DataFrame = unpivot_matches(matchday, index)

    matches_by_team: pl.DataFrame = unpivot_matches(matches)

    team_features_matchday: pl.DataFrame = (
        extract_matchday_team_features(matchday_by_team, matches_by_team)
    )

    matchday_features: pl.DataFrame = (
        team_features_matchday
            .pivot(on='location', index=index)
    )

    return matchday_features


def extract_matchday_team_features(
        matchday_by_team: pl.DataFrame,
        matches_by_team: pl.DataFrame
) -> pl.DataFrame:
    form_features: pl.DataFrame = (
        matchday_by_team
            .join(matches_by_team, on='team', how='left')
            .sort('date_right')
            .group_by('team')
            .agg(
                pl.col('goals_for', 'goals_against', 'win', 'loss')
                    .tail(10)
                    .mean()
                    .name.prefix('form_avg_'),
                pl.col('win', 'loss')
                    .filter(pl.col('tournament_right') == 'World Cup')
                    .tail(10)
                    .mean()
                    .name.map(lambda name: f'form_avg_{name}_wc'),
                pl.col('win', 'loss')
                    .filter(pl.col('tournament_right') == 'Confederation Cup')
                    .tail(10)
                    .mean()
                    .name.map(lambda name: f'form_avg_{name}_cc'),
            )
            .with_columns(
                pl.col([
                    f'form_avg_{result}_{tournament}'
                    for result in ('win', 'loss')
                    for tournament in ('wc', 'cc')
                ]).fill_null(0)
            )
    )

    h2h_features: pl.DataFrame = (
        matchday_by_team
            .join(matches_by_team, on=['team', 'opponent'], how='left')
            .sort('date_right')
            .group_by('team', 'opponent')
            .agg(
                pl.col('win', 'loss').mean().name.prefix('h2h_rate_')
            )
            .with_columns(
                pl.col('h2h_rate_win', 'h2h_rate_loss').fill_null(0)
            )
    )

    team_features_matchday: pl.DataFrame = (
        matchday_by_team
            .join(form_features, on='team', how='left')
            .join(h2h_features, on=['team', 'opponent'], how='left')
    )

    return team_features_matchday


def predict_matchday(
        matchday_features: pl.DataFrame, 
        models: dict[str, Pipeline],
        rng: np.random.Generator
) -> pl.DataFrame:
    classes: np.ndarray = models['results'].classes_

    results_predicted: np.ndarray = np.array([
        rng.choice(classes, p=distribution)
        for distribution in models['results'].predict_proba(matchday_features)
    ])

    scores_predicted: np.ndarray = rng.poisson(
        np.clip(models['goals'].predict(matchday_features), 0, 10)
    )

    matchday_scores: pl.DataFrame = (
        matchday_features
            .with_columns(
                result_home=pl.Series(results_predicted),
                score_home=pl.Series(scores_predicted[:, 0]),
                score_away=pl.Series(scores_predicted[:, 1])
            )
            .with_columns(
                score_away=pl.when(
                    pl.col('score_home').ge(pl.col('score_away'))
                    & pl.col('result_home').eq(pl.lit('loss'))
                )
                    .then(pl.col('score_home') + 1)
                    .otherwise(pl.col('score_away')),
                score_home=pl.when(
                    pl.col('score_home').le(pl.col('score_away'))
                    & pl.col('result_home').eq(pl.lit('win'))
                )
                    .then(pl.col('score_away') + 1)
                    .when(
                        pl.col('score_home').ne(pl.col('score_away')) 
                        & pl.col('result_home').eq(pl.lit('draw'))
                    )
                    .then(pl.col('score_away'))
                    .otherwise(pl.col('score_home'))
            )
    )

    predictions_matchday: pl.DataFrame = process_match_scores(matchday_scores)

    return predictions_matchday


def get_group_table(
        predictions_group_stage: list[pl.DataFrame]
) -> pl.DataFrame:
    group_table: pl.DataFrame = (
        unpivot_matches(
            pl.concat(predictions_group_stage), 
            index=['match_id', 'date', 'group_name']
        )
        .with_columns(
            points=pl
                .when(pl.col('win'))
                .then(3)
                .when(pl.col('loss'))
                .then(0)
                .otherwise(1),
            gd=pl.col('goals_for') - pl.col('goals_against')
        )
        .group_by('group_name', 'team')
        .agg(
            pl.col('points', 'gd', 'goals_for').sum()
        )
        .sort(
            ['group_name', 'points', 'gd', 'goals_for'],
            descending=[False, True, True, True]
        )
    )

    return group_table


def create_bracket(group_table: pl.DataFrame) -> pl.DataFrame:
    top_from_groups: pl.DataFrame = (
        group_table
            .group_by('group_name')
            .head(2)
            .with_columns(
                position=pl.int_range(pl.len())
                    .add(1)
                    .over('group_name')
                    .cast(pl.Int32)
            )
    )

    thirds_from_groups: pl.DataFrame = (
        group_table
            .group_by('group_name')
            .tail(2)
            .group_by('group_name')
            .head(1)
            .sort(['points', 'gd', 'goals_for'], descending=True)
            .head(8)
            .with_columns(position=3)
    )

    teams_qualified_knockouts: pl.DataFrame = pl.concat(
        [top_from_groups, thirds_from_groups]
    )

    three_team_groups: pl.Series = thirds_from_groups.get_column('group_name')

    bracket_template: pl.DataFrame = (
        pl.from_records(
            [
                ('e', 1), (three_team_groups[0], 3),
                ('i', 1), (three_team_groups[1], 3),
                ('a', 2), ('b', 2),
                ('f', 1), ('c', 2),
                ('k', 2), ('l', 2),
                ('h', 1), ('j', 2),
                ('d', 1), (three_team_groups[2], 3),
                ('g', 1), (three_team_groups[3], 3),
                ('c', 1), ('f', 2),
                ('e', 2), ('i', 2),
                ('a', 1), (three_team_groups[4], 3),
                ('l', 1), (three_team_groups[5], 3),
                ('j', 1), ('h', 2),
                ('d', 2), ('g', 2),
                ('b', 1), (three_team_groups[6], 3),
                ('k', 1), (three_team_groups[7], 3)
            ],
            schema=['group_name', 'position'],
            orient='row'
        )
        .with_columns(match_id=pl.int_range(pl.len()) // 2)
        .select('match_id', 'group_name', 'position')
    )
    
    bracket_by_team: pl.DataFrame = (
        bracket_template
            .join(
                teams_qualified_knockouts
                    .select('group_name', 'position', 'team'),
                on=['group_name', 'position'],
                how='left'
            )
            .select('match_id', 'team')
    )

    bracket: pl.DataFrame = build_bracket_from_teams(bracket_by_team)
    
    return bracket


def build_bracket_from_teams(
        bracket_by_team: pl.DataFrame,
        host_nations: list[str] = ['United States', 'Mexico', 'Canada']
) -> pl.DataFrame:
    bracket: pl.DataFrame = (
        bracket_by_team
            .with_columns(
                team_number=pl.int_range(pl.len()).over('match_id')
            )
            .pivot('team_number', index='match_id')
            .with_columns(
                team_home=pl.when(pl.col('1').is_in(host_nations))
                    .then(pl.col('1'))
                    .otherwise(pl.col('0'))
            )
            .with_columns(
                team_away=pl.when(pl.col('0') == pl.col('team_home'))
                    .then(pl.col('1'))
                    .otherwise(pl.col('0'))
            )
            .select('team_home', 'team_away')
    )

    return bracket


def predict_knockout_stage(
        bracket: pl.DataFrame,
        models: dict[str, Pipeline],
        elos: pl.DataFrame,
        matches: pl.DataFrame,
        rng: np.random.Generator,
        predictions_so_far: list[pl.DataFrame] = [],
        start_date: pl.Expr = pl.date(2026, 6, 25)
) -> list[pl.DataFrame]:
    matchday: pl.DataFrame = get_matchday(
        bracket, 
        elos,
        start_date,
        stage='knockout'
    )

    predictions_matchday, matches_updated, elos_updated = (
        predict_matchday_update_records(
            matchday,
            matches,
            elos,
            models,
            rng
        )
    )

    draws: list[int] = locate_draws(predictions_matchday)

    predictions_post_shootout: pl.DataFrame = predict_shootouts(
        predictions_matchday, 
        draws, 
        rng
    )

    if (len(predictions_post_shootout) == 1):
        return (predictions_so_far + [predictions_post_shootout])

    bracket_by_team_next_round: pl.DataFrame = get_next_round(
        predictions_post_shootout
    )

    bracket_next_round = build_bracket_from_teams(bracket_by_team_next_round)

    return predict_knockout_stage(
        bracket_next_round,
        models,
        elos_updated,
        matches_updated,
        rng,
        predictions_so_far + [predictions_post_shootout],
        start_date.add(pl.duration(days=1))
    )


def locate_draws(predictions_matchday: pl.DataFrame) -> list[int]:
    draws: list[int] = (
        predictions_matchday
            .filter(
                (pl.col('win_home').not_()) & (pl.col('loss_home').not_())
            )
            .get_column('match_id')
            .to_list()
    )

    return draws


def predict_shootouts(
        predictions_matchday: pl.DataFrame, 
        draws: list[int], 
        rng: np.random.Generator
) -> pl.DataFrame:
    samples: np.ndarray = rng.uniform(size=len(draws))

    shootout_wins_home: list[int] = [
        draw for draw, sample in zip(draws, samples) if sample >= 0.5
    ]

    predictions_post_shootout: pl.DataFrame = (
        predictions_matchday
            .with_columns(
                win_home=pl.when(
                    pl.col('match_id').is_in(shootout_wins_home)
                )
                    .then(True)
                    .otherwise(pl.col('win_home')),
                win_away=pl.when(
                    (pl.col('match_id').is_in(draws)) 
                    & (~pl.col('match_id').is_in(shootout_wins_home))
                )
                    .then(True)
                    .otherwise(pl.col('win_away'))
            )
    )

    return predictions_post_shootout


def get_next_round(predictions_post_shootout: pl.DataFrame) -> pl.DataFrame:
    bracket_teams_next_round: pl.DataFrame = (
        unpivot_matches(predictions_post_shootout)
            .filter(pl.col('win'))
            .with_columns(
                match_id=pl.int_range(pl.len()) // 2
            )
            .select('match_id', 'team')
    )

    return bracket_teams_next_round


def get_winner(predictions_knockout_stage: list[pl.DataFrame]) -> str:
    winner: str = (
        unpivot_matches(predictions_knockout_stage[-1])
            .filter(pl.col('win'))
            .get_column('team')
            .item()
    )

    return winner