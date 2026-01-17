import polars as pl
from typing import Any
from itertools import chain
from collections import Counter
import copy
from soccer.clean_data import prepare_matches


def get_qualified_teams(
        already_qualified: list[str],
        elos: pl.DataFrame,
        team_confederations: pl.DataFrame
) -> pl.DataFrame:
    counts_qualified_by_confederation: pl.DataFrame = (
        pl.DataFrame({'team': already_qualified})
            .join(team_confederations, on='team', how='left')
            .group_by('confederation')
            .agg(qualified=pl.len())
    )

    slots_by_confederation: pl.DataFrame = (
        pl.DataFrame(
            {
                'confederation': 
                    ['UEFA', 'CAF', 'AFC', 'CONMEBOL', 'CONCACAF', 'OFC'],
                'slots': [16, 9, 8, 6, 6, 1]
            }
        )
        .join(
            counts_qualified_by_confederation,
            on='confederation',
            how='left'
        )
        .with_columns(pl.col('qualified').fill_null(0))
        .with_columns(remaining=pl.col('slots') - pl.col('qualified'))
    )

    elos_with_confederation: pl.DataFrame = (
        elos.join(team_confederations, on='team', how='left')
    )

    teams_qualified_standard: list[str] = list(
        chain.from_iterable(
            get_confederation_qualified_teams(
                confederation_dict, 
                elos_with_confederation, 
                already_qualified   
            )
            for confederation_dict in slots_by_confederation.to_dicts()
        )
    )

    teams_qualified_playoff: list[str] = (
        elos_with_confederation
            .filter(
                ~pl.col('confederation').is_in(['UEFA', 'Other']),
                ~pl.col('team').is_in(
                    already_qualified + teams_qualified_standard
                )
            )
            .top_k(2, by='elo')
            .get_column('team')
            .to_list()
    )

    teams_qualified: pl.DataFrame = (
        elos_with_confederation
            .filter(
                pl.col('team').is_in(
                    already_qualified 
                    + teams_qualified_standard 
                    + teams_qualified_playoff
                )
            )
    )

    return teams_qualified


def get_confederation_qualified_teams(
        confederation_dict: dict[str, Any],
        elos_with_confederation: pl.DataFrame,
        already_qualified: list[str]
) -> pl.Series:
    confederation: str = confederation_dict['confederation']

    remaining: int = confederation_dict['remaining']

    confederation_qualified_teams: pl.Series = (
        elos_with_confederation
            .filter(
                pl.col('confederation') == confederation,
                ~pl.col('team').is_in(already_qualified)
            )
            .top_k(remaining, by='elo')
            .get_column('team')
    )

    return confederation_qualified_teams


def simulate_group_draw(
        teams_qualified: pl.DataFrame,
        already_drawn: dict[str, list[str]],
        host_nations: list[str] = ['United States', 'Mexico', 'Canada']
) -> pl.DataFrame:
    teams_sorted: pl.DataFrame = (
        teams_qualified
            .sort(
                pl.col('team').is_in(host_nations), 
                'elo', 
                descending=[True, True]
            )
    )

    pots: list[pl.DataFrame] = [
        teams_sorted.slice(i, 12) for i in range(0, len(teams_sorted), 12)
    ]

    groups_drawn: dict[str, list[dict[str, Any]]] = get_initial_groups(
        already_drawn,
        teams_sorted
    )

    pots_of_available_teams: list[pl.DataFrame] = pots.copy()

    for group_name in groups_drawn.keys():
        groups_drawn, pots_of_available_teams = draw_group(
            group_name,
            groups_drawn,
            pots_of_available_teams
        )

    groups: pl.DataFrame = pl.DataFrame({
        'group_name': groups_drawn.keys(),
        'teams': [
            [team['team'] for team in group]
            for group in groups_drawn.values()
        ]
    })

    return groups


def get_initial_groups(
        already_drawn: dict[str, list[str]],
        teams_sorted: pl.DataFrame
) -> dict[str, list[dict[str, Any]]]:
    group_names: str = 'abcdefghijkl'

    groups_drawn: dict[str, list[dict[str, Any]]] = {
        k: teams_sorted
            .filter(pl.col('team').is_in(already_drawn[k]))
            .to_dicts() if k in already_drawn else []
        for k in group_names
    }

    return groups_drawn


def draw_group(
    group_name: str,
    groups_drawn: dict[str, list[dict[str, Any]]],
    pots_of_available_teams: list[pl.DataFrame]
) -> tuple[dict[str, list[dict[str, Any]]], list[pl.DataFrame]]:
    group: list[dict[str, Any]] = groups_drawn[group_name]

    pots_of_available_teams_updated: list[pl.DataFrame] = (
        pots_of_available_teams.copy()
    )

    for pot_idx, pot in enumerate(pots_of_available_teams):
        group, remaining_from_pot, groups_drawn = process_pot(
            pot, 
            pot_idx,
            group_name,
            group, 
            groups_drawn
        )

        pots_of_available_teams_updated[pot_idx] = remaining_from_pot

    groups_drawn_updated: dict[str, list[dict[str, Any]]] = (
        copy.deepcopy(groups_drawn)
    )
    
    groups_drawn_updated[group_name] = group

    return groups_drawn_updated, pots_of_available_teams_updated


def process_pot(
    pot: pl.DataFrame,
    pot_idx: int,
    group_name: str,
    group: list[dict[str, Any]],
    groups_drawn: dict[str, list[dict[str, Any]]]
) -> tuple[
    list[dict[str, Any]], 
    pl.DataFrame, 
    dict[str, list[dict[str, Any]]]
]:
    if len(group) > pot_idx:
        team: dict[str, Any] | None = group[pot_idx]

        remaining_from_pot: pl.DataFrame = (
            pot.filter(pl.col('team') != team['team'])
        )

        return group, remaining_from_pot, groups_drawn

    shuffled_pot: list[dict[str, Any]] = (
        pot.sample(fraction=1, shuffle=True, seed=42).to_dicts()
    )

    team, remaining_from_pot = draw_team_from_pot(
        shuffled_pot,
        group
    )

    if team is None:
        for team_to_swap_in in shuffled_pot:
            for other_group_name, other_group in groups_drawn.items():
                team, remaining_from_pot, groups_drawn = try_swap(
                    team_to_swap_in,
                    other_group,
                    other_group_name,
                    group,
                    shuffled_pot,
                    pot_idx,
                    groups_drawn
                )

                if team is not None:
                    group = group + [team]

                    return group, remaining_from_pot, groups_drawn

        raise ValueError(
            'No valid group configuration found for' 
            + f' group {group_name} pot {pot_idx+1}'
        )
    
    group = group + [team]

    return group, remaining_from_pot, groups_drawn


def draw_team_from_pot(
    shuffled_pot: list[dict[str, Any]],
    group: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, pl.DataFrame]:
    for team in shuffled_pot:
        group_updated: list[dict[str, Any]] = group + [team]

        if is_valid_partial_group(group_updated):
            remaining_from_pot: pl.DataFrame = pl.DataFrame(
                [t for t in shuffled_pot if t != team]
            )

            return team, remaining_from_pot
    
    return None, pl.DataFrame(shuffled_pot)


def try_swap(
    team_to_swap_in: dict[str, Any],
    other_group: list[dict[str, Any]],
    other_group_name: str,
    group: list[dict[str, Any]],
    shuffled_pot: list[dict[str, Any]],
    pot_idx: int,
    groups_drawn: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, Any] | None, pl.DataFrame, dict[str, list[dict[str, Any]]]]:
    groups_drawn_updated: dict[str, list[dict[str, Any]]] = (
        copy.deepcopy(groups_drawn)
    )

    team_to_swap_out: dict[str, Any] = other_group[pot_idx]

    group_current_is_valid: bool = is_valid_partial_group(
        group + [team_to_swap_out]
    )

    group_with_swap_is_valid: bool = is_valid_partial_group(
        other_group[:pot_idx] 
        + [team_to_swap_in] 
        + other_group[pot_idx+1:]
    )

    if group_current_is_valid and group_with_swap_is_valid:
        team: dict[str, Any] = team_to_swap_out

        remaining_from_pot: pl.DataFrame = pl.DataFrame(
            [t for t in shuffled_pot if t != team_to_swap_in]
        )

        groups_drawn_updated[other_group_name][pot_idx] = team_to_swap_in

        return team, remaining_from_pot, groups_drawn_updated

    return None, pl.DataFrame(shuffled_pot), groups_drawn


def is_valid_partial_group(group: list[dict[str, Any]]) -> bool:
    conf_list: list[str | None] = [
        team['confederation'] for team in group
    ]

    conf_counts: Counter[str] = Counter(
        conf for conf in conf_list if conf is not None
    )
    
    uefa_is_valid: bool = conf_counts.get('UEFA', 0) <= 2
    
    other_confs_are_valid: bool = all(
        count <= 1 
        for conf, count in conf_counts.items() 
        if conf != 'UEFA'
    )

    is_valid: bool = uefa_is_valid and other_confs_are_valid

    return is_valid


def get_group_stage_matchdays(
        groups: pl.DataFrame,
        elos: pl.DataFrame
) -> list[pl.DataFrame]:
    matchup_patterns: list[list[tuple[int, ...]]] = get_matchup_patterns(4)

    matchups_group_stage: list[pl.DataFrame] = [
        pl.concat(
            get_group_matchups(group, matchup_pattern)
            for group in groups.to_dicts()
        )
        for matchup_pattern in matchup_patterns
    ]

    matchday_dates: list[pl.Expr] = [
        pl.date(2026, 6, 11 + i) for i in range(3)
    ]

    matchdays_group_stage: list[pl.DataFrame] = [
        get_matchday(
            matchups, 
            elos,
            matchday_date,
            stage='group'
        ) 
        for matchups, matchday_date in zip(
            matchups_group_stage, 
            matchday_dates
        )
    ]

    return matchdays_group_stage


def get_matchup_patterns(n_teams: int) -> list[list[tuple[int, ...]]]:
    all_team_indices: set[int] = set(range(n_teams))

    matchup_patterns: list[list[tuple[int, ...]]] = [
        [(0, opp_idx), tuple(all_team_indices - {0, opp_idx})] 
        for opp_idx in range(1, n_teams)
    ]

    return matchup_patterns


def get_group_matchups(
        group: dict[str, Any],
        matchup_pattern: list[tuple[int, ...]]
) -> pl.DataFrame:
    group_name, teams = group.values()

    matchups: pl.DataFrame = (
        pl.DataFrame(
            [
                [teams[home_idx], teams[away_idx]]
                for home_idx, away_idx in zip(*matchup_pattern)
            ],
            schema=['team_home', 'team_away']
        )
        .with_columns(group_name=pl.lit(group_name))
    )

    return matchups


def get_matchday(
        matchups: pl.DataFrame, 
        elos: pl.DataFrame,
        matchday_date: pl.Expr,
        stage: str,
        host_nations: list[str] = ['United States', 'Mexico', 'Canada']
) -> pl.DataFrame:
    columns_to_select: list[str] = [
        'date', 'group_name', 'team_home', 'team_away', 
        'tournament', 'country', 'neutral'
    ]

    if stage == 'knockout':
        columns_to_select = [
            column for column in columns_to_select if column != 'group_name'
        ]

    matchday: pl.DataFrame = (
        matchups
        .with_columns(
            date=matchday_date,
            tournament=pl.lit('World Cup'),
            country=pl.when(pl.col('team_home').is_in(host_nations))
                .then(pl.col('team_home'))
                .otherwise(pl.lit('United States'))
                .alias('country'),
        )
        .with_columns(
            neutral=(pl.col('team_home') != pl.col('country'))
        )
        .select(columns_to_select)
        .pipe(prepare_matches)
        .join(elos, left_on='team_home', right_on='team', how='left')
        .rename({'elo': 'elo_home'})
        .join(elos, left_on='team_away', right_on='team', how='left')
        .rename({'elo': 'elo_away'})
    )

    return matchday