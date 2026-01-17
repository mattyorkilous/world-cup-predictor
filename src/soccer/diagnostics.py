import polars as pl
from extract_features import unpivot_matches


def get_top_teams_by_decade(matches_with_elos: pl.DataFrame) -> pl.DataFrame:
    matches_with_decade: pl.DataFrame = (
        matches_with_elos
            .with_columns(
                decade=pl.col('date').cast(pl.Date).dt.year() // 10 * 10
            )
    )

    matches_with_decade_by_team: pl.DataFrame = unpivot_matches(
        matches_with_decade,
        index=['match_id', 'date', 'decade']
    )

    top_teams_by_decade_before_today: pl.DataFrame = (
        matches_with_decade_by_team
            .sort('date')
            .group_by(['decade', 'team'], maintain_order=True)
            .first()
            .with_columns(
                rank=pl.col('elo')
                    .rank('ordinal', descending=True)
                    .over('decade')
            )
            .filter((pl.col('rank') <= 3) & (pl.col('decade') >= 1930))
            .select(['decade', 'rank', 'team', 'elo'])
            .sort(['decade', 'rank'])
    )

    top_teams_today: pl.DataFrame = (
        matches_with_decade_by_team
            .sort('date', descending=True)
            .group_by('team', maintain_order=True)
            .first()
            .sort('elo', descending=True)
            .with_columns(
                decade=2025
            )
            .with_columns(
                rank=pl.col('elo').rank('ordinal', descending=True)
            )
            .filter(pl.col('rank') <= 3)
            .select(['decade', 'rank', 'team', 'elo'])
    )

    top_teams_by_decade: pl.DataFrame = (
        pl.concat([
            top_teams_by_decade_before_today,
            top_teams_today
        ])
    )

    return top_teams_by_decade
