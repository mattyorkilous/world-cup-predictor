import polars as pl
import polars.selectors as cs


def extract_match_features(matches_with_elos: pl.DataFrame) -> pl.DataFrame:
    matches_by_team: pl.DataFrame = unpivot_matches(matches_with_elos)

    team_match_features: pl.DataFrame = extract_team_match_features(
        matches_by_team
    )

    matches_with_features: pl.DataFrame = join_team_match_features(
        matches_with_elos, 
        team_match_features
    )

    return matches_with_features


def unpivot_matches(
        matches_with_elos: pl.DataFrame, 
        index: list[str] = ['match_id', 'date', 'tournament']
) -> pl.DataFrame:
    columns_to_unpivot: list[str] = [
        col for col in matches_with_elos.columns 
        if '_home' in col or '_away' in col
    ]
    
    dtypes_of_unpivoted_columns: dict[str, pl.DataType] = {
        col.replace('_home', ''): matches_with_elos.schema[col]
        for col in columns_to_unpivot
        if '_home' in col
    }

    dtypes_of_castable_columns: dict[str, pl.DataType] = {
        k: v for k, v in dtypes_of_unpivoted_columns.items()
        if v not in (pl.Boolean, pl.String)
    }

    boolean_columns: list[str] = [
        k for k, v in dtypes_of_unpivoted_columns.items()
        if v == pl.Boolean
    ]
    
    matches_by_team: pl.DataFrame = (
        matches_with_elos
            .unpivot(columns_to_unpivot, index=index)
            .with_columns(
                pl.col('variable')
                    .str.extract_groups('(.+)_(home|away)')
                    .struct.rename_fields(['var_name', 'location'])
            )
            .unnest('variable')
            .pivot(
                'var_name',
                index=(index + ['location']),
                values='value'
            )
            .with_columns(
                pl.col(col).cast(dtypes_of_castable_columns[col])
                for col in dtypes_of_castable_columns
            )
            .with_columns(
                pl.col(boolean_columns)
                    .replace_strict({'true': True, 'false': False})
            )
    )
    
    return matches_by_team


def extract_team_match_features(matches_by_team: pl.DataFrame) -> pl.DataFrame:
    team_tournament_form_stats: pl.DataFrame = (
        get_team_tournament_form_stats(matches_by_team)
    )

    team_match_features: pl.DataFrame = (
        matches_by_team
            .with_columns(
                get_avg_last_10(
                    pl.col('win', 'loss', 'goals_for', 'goals_against')
                )
                    .over('team', order_by='date')
                    .name.prefix('form_avg_')
            )
            .with_columns(
                get_cumulative_avg(pl.col('win', 'loss'))
                    .over(['team', 'opponent'], order_by='date')
                    .name.prefix('h2h_rate_')
            )
            .join(
                team_tournament_form_stats,
                on=['match_id', 'team'],
                how='left'
            )
            .with_columns(
                pl.col([
                    f'form_avg_{result}_{tourn}' 
                    for result in ('win', 'loss')
                    for tourn in ('wc', 'cc')
                ])
                    .forward_fill()
                    .over('team', order_by='date')
                    .fill_null(0)
            )
    )

    return team_match_features


def get_team_tournament_form_stats(
        matches_by_team: pl.DataFrame
) -> pl.DataFrame:
    team_tournament_form_stats: pl.DataFrame = (
        matches_by_team
            .filter(
                pl.col('tournament').is_in(['World Cup', 'Confederation Cup'])
            )
            .with_columns(
                get_avg_last_10(pl.col('win', 'loss'))
                    .over(['team', 'tournament'], order_by='date')
                    .name.prefix('form_avg_')
            )
            .select(
                'match_id',
                pl.col('tournament').replace(
                    {
                        'World Cup': 'wc',
                        'Confederation Cup': 'cc'
                    }
                ),
                'team',
                cs.starts_with('form_avg_')
            )
            .pivot(
                'tournament',
                index=['match_id', 'team'],
                values=cs.starts_with('form_avg_')
            )
    )

    return team_tournament_form_stats


def get_avg_last_10(x: pl.Expr) -> pl.Expr:
    avg_last_10: pl.Expr = (
        x
            .shift(1)
            .rolling_mean(10, min_samples=1)
            .fill_null(0)
    )

    return avg_last_10


def get_cumulative_avg(x: pl.Expr) -> pl.Expr:
    cumulative_avg: pl.Expr = (
        x
            .shift(1)
            .cum_sum()
            .fill_null(0)
            .truediv(
                x.cum_count().sub(1)
            )
            .fill_nan(0)
    )

    return cumulative_avg


def join_team_match_features(
        matches_with_elos: pl.DataFrame, 
        team_match_features: pl.DataFrame
) -> pl.DataFrame:
    values: list[str] = [
        col for col in team_match_features.columns
        if 'form_avg_' in col or 'h2h_rate_' in col
    ]

    features: pl.DataFrame = (
        team_match_features
            .pivot(
                'location',
                index='match_id',
                values=values
            )
    )
    
    matches_with_features: pl.DataFrame = matches_with_elos.join(
        features, 
        on='match_id',
        how='left'
    )
    
    return matches_with_features