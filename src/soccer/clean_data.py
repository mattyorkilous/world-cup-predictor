import polars as pl
from polars import selectors as cs


def clean_matches(datasets_raw: dict[str, pl.DataFrame]) -> pl.DataFrame:
    countries, matches_raw = [
        datasets_raw[f'{key}.csv'] 
        for key in ('countries_names', 'all_matches')
    ]

    countries_dict: dict[str, str] = dict(
        countries.select(cs.ends_with('name')).iter_rows()
    )

    matches_clean_names: pl.DataFrame = clean_country_names(
        matches_raw, 
        countries_dict
    )

    matches: pl.DataFrame = (
        prepare_matches(matches_clean_names)
            .pipe(process_match_scores)
    )

    return matches


def clean_country_names(
        matches_raw: pl.DataFrame, 
        countries_dict: dict[str, str]
) -> pl.DataFrame:
    matches_clean_names: pl.DataFrame = (
        matches_raw
            .with_columns(
                pl.col('home_team', 'away_team', 'country')
                    .replace(
                        'São Tomé and Príncipe',
                        'Sao Tome and Principe'
                    )
                    .replace(countries_dict)
            )
            .rename(swap_underscore)
    )

    return matches_clean_names


def prepare_matches(matches_clean_names: pl.DataFrame) -> pl.DataFrame:
    matches_prepared: pl.DataFrame = (
        matches_clean_names
            .with_row_index('match_id')
            .with_columns(
                pl.col('date').cast(pl.Date)
            )
            .with_columns(
                tournament=classify_tournament(pl.col('tournament'))
            )
            .with_columns(
                opponent_home=pl.col('team_away'),
                opponent_away=pl.col('team_home'),
                importance=pl.col('tournament')
                    .replace_strict(
                        {
                            'World Cup': 55,
                            'Confederation Cup': 40,
                            'Qualifiers': 25,
                            'Nations League': 15,
                            'Friendly': 10
                        }
                    )
            )
    )

    return matches_prepared


def process_match_scores(matches_prepared: pl.DataFrame) -> pl.DataFrame:
    matches: pl.DataFrame = (
        matches_prepared
            .with_columns(
                truncate_score(pl.col('score_home', 'score_away')).name.keep()
            )
            .with_columns(
                result_home=get_match_result(
                    pl.col('score_home'), 
                    pl.col('score_away')
                ),
                result_away=get_match_result(
                    pl.col('score_away'), 
                    pl.col('score_home')
                ),
                goals_against_home=pl.col('score_away'),
                goals_against_away=pl.col('score_home'),
                margin=pl.col('score_home') - pl.col('score_away')
            )
            .with_columns(
                is_result_location(result, location)
                for result in ('win', 'loss') 
                for location in ('home', 'away')
            )
            .with_columns(
                margin_multiplier=pl.col('margin')
                    .replace_strict(
                        {0: 1.0, 1: 1.25, 2: 1.5, 3: 1.75}, 
                        default=2.0
                    )
            )
            .rename({
                'score_home': 'goals_for_home', 
                'score_away': 'goals_for_away'
            })
    )

    return matches


def swap_underscore(text: str) -> str:
    if '_' in text:
        first, second = text.split('_', 1)

        swapped: str = f"{second}_{first}"

        return swapped
    
    return text


def truncate_score(score: pl.Expr) -> pl.Expr:
    score_truncated: pl.Expr = (
        pl.when(score >= 10)
            .then(10)
            .otherwise(score)
    )

    return score_truncated


def classify_tournament(tournament: pl.Expr) -> pl.Expr:
    is_world_cup: pl.Expr = (tournament == 'World Cup')

    is_confederation_cup: pl.Expr = tournament.is_in([
        'Intercontinental Champ',
        'Artemio Franchi Trophy',
        'Asian Cup',
        'African Nations Cup',
        'Copa America',
        'Copa América',
        'CONCACAF Championship',
        'European Championship',
        'Confederations Cup',
        'Confederation Cup',
        'AFC-OFC Challenge Cup',
        'Afro-Asian Cup',
        'Oceania Nations Cup',
        'South American Champ',
        'Panamerican Championship',
        'CONCACAF Cup'
    ])

    is_qualifier: pl.Expr = tournament.is_in([
        'World Cup qualifier',
        'WC q and Oce Cup',
        'WC q and CONCACAF Ch',
        'WC q & British Ch',
        'WC q & Nordic Ch',
        'WC and African Cup qual',
        'WC and Asian Cup qual',
        'WC and CONCACAF Ch q',
        'Asian Cup qualifier',
        'African Nations Cup qualifier',
        'Copa América qualifier',
        'CONCACAF Champ qual',
        'European Championship qual',
        'Oceania Nations Cup qualifier',
        'Asian Cup q & Asian Chlg Cup',
        'Asian Cup & Asian Chlg Cup q',
        'CONCACAF Ch q & Car Ch',
        'CONCACAF Ch q & Car Cup',
        'CONCACAF Ch q & C Am Cup',
        'CONCACAF Ch q & Car Ch PO',
        'Pan American Games qualifier',
        'World Cup and African Cup qual',
        'World Cup and Asian Cup qual',
        'WC and Oce Cup q',
        'WC and Oce Cup q and S Pac G'
    ])
    
    is_nations_league: pl.Expr = tournament.is_in([
        'European Nations League',
        'European Nations League A',
        'European Nations League B',
        'European Nations League C',
        'European Nations League D',
        'CONCACAF Nations League',
        'CONCACAF Nations League A',
        'CONCACAF Nations League B',
        'CONCACAF Nations League C',
        'Nations League',
        'European Nations League A/B',
        'European Nations League B/C',
        'CONCACAF Nations League q'
    ])

    tournament_classified: pl.Expr = (
        pl.when(is_world_cup)
            .then(pl.lit('World Cup'))
            .when(is_confederation_cup)
            .then(pl.lit('Confederation Cup'))
            .when(is_qualifier)
            .then(pl.lit('Qualifiers'))
            .when(is_nations_league)
            .then(pl.lit('Nations League'))
            .otherwise(pl.lit('Friendly'))
    )

    return tournament_classified


def get_match_result(
        goals_team: pl.Expr, 
        goals_opponent: pl.Expr
) -> pl.Expr:
    match_result: pl.Expr = (
        pl.when(goals_team > goals_opponent)
            .then(pl.lit('win'))
            .when(goals_team == goals_opponent)
            .then(pl.lit('draw'))
            .otherwise(pl.lit('loss'))
    )

    return match_result


def is_result_location(result: str, location: str) -> pl.Expr:
    is_result_location: pl.Expr = (
        (pl.col(f'result_{location}') == result)
            .alias(f'{result}_{location}')
    )

    return is_result_location