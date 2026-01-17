import polars as pl
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import PoissonRegressor, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from soccer.clean_data import get_match_result


def select_model(
        matches_with_features: pl.DataFrame, 
) -> tuple[dict[str, Pipeline], pl.DataFrame]:
    (X_train, X_test, 
     y_train_results, y_test_results, 
     y_train_goals,
     sample_weight_train) = split_data(matches_with_features)
    
    estimators: dict[str, BaseEstimator] = {
        'logistic_regression': LogisticRegression(max_iter=1000),
        'random_forest': RandomForestClassifier(n_jobs=-1),
        'neural_network': MLPClassifier(max_iter=1000)
    }

    models_results: dict[str, Pipeline] = {
        model_name: train_model(
            estimator, 
            X_train, 
            y_train_results, 
            sample_weight_train
        )
        for model_name, estimator in estimators.items()
    }

    model_goals: Pipeline = train_model(
        MultiOutputRegressor(PoissonRegressor(max_iter=1000)),
        X_train,
        y_train_goals,
        sample_weight_train
    )

    accuracy_summary: pl.DataFrame = get_accuracies_table(
        models_results,
        X_test,
        y_test_results
    )

    model_results: Pipeline = models_results[
        str(accuracy_summary.get_column('model').first())
    ]

    models: dict[str, Pipeline] = { 
        'results': model_results,
        'goals': model_goals
    }

    return models, accuracy_summary


def split_data(
        matches_with_features: pl.DataFrame
) -> list[pl.DataFrame | pl.Series]:
    features: list[str] = (
        ['tournament']
        + [
            f'{var}_{team_type}' 
            for var in [
                'elo',
                'form_avg_goals_for', 
                'form_avg_goals_against',
                'form_avg_win',
                'form_avg_loss',
                'form_avg_win_wc',
                'form_avg_loss_wc',
                'form_avg_win_cc',
                'form_avg_loss_cc'
            ]
            for team_type in ('home', 'away')
        ]
        + [
            'h2h_rate_win_home',
            'h2h_rate_loss_home',
            'neutral'
        ]
    )

    labels_goals: list[str] = ['goals_for_home', 'goals_for_away']

    X: pl.DataFrame = matches_with_features.select(features)
    
    y_goals: pl.DataFrame = matches_with_features.select(labels_goals)

    y_results: pl.Series = matches_with_features.get_column('result_home')

    sample_weight: pl.Series = matches_with_features.get_column('importance')

    (X_train, X_test, 
     y_train_results, y_test_results, 
     y_train_goals, _, 
     sample_weight_train, _) = (
        train_test_split(X, y_results, y_goals, sample_weight, random_state=42)
    )

    return [
        X_train, X_test, 
        y_train_results, y_test_results, 
        y_train_goals, 
        sample_weight_train
    ]


def train_model(
        estimator: BaseEstimator,
        X_train: pl.DataFrame | pl.Series, 
        y_train: pl.DataFrame | pl.Series,
        sample_weight_train: pl.DataFrame | pl.Series
) -> Pipeline:
    column_transformer: ColumnTransformer = ColumnTransformer([
        (
            'categorical', 
            OneHotEncoder(), 
            [
                col 
                for col, dtype in X_train.schema.items() # type: ignore
                if dtype == pl.Utf8
            ]
        ),
        (
            'numeric', 
            StandardScaler(), 
            [
                col 
                for col, dtype in X_train.schema.items() # type: ignore
                if dtype != pl.Utf8
            ]
        )
    ])

    pipeline: Pipeline = Pipeline(
        [
            ('tfrm_cols', column_transformer),
            ('model', estimator)
        ]
    )

    if hasattr(estimator, 'sample_weight'):
        pipeline.fit(
            X_train,  # type: ignore
            y_train,  # type: ignore
            model__sample_weight=sample_weight_train  # type: ignore
        )
    else:
        pipeline.fit(
            X_train,  # type: ignore
            y_train,  # type: ignore
        )

    model: Pipeline = pipeline

    return model


def get_accuracies_table(
        models_results: dict[str, Pipeline],
        X_test: pl.DataFrame | pl.Series,
        y_test_results: pl.DataFrame | pl.Series
) -> pl.DataFrame:
    model_predictions: dict[str, np.ndarray] = {
        model_name: model.predict(X_test) # type: ignore
        for model_name, model in models_results.items()
    }

    predictions_elo: pl.Series = (
        X_test
            .select( # type: ignore
                result_home=get_match_result(
                    pl.col('elo_home'), 
                    pl.col('elo_away')
                )
            )
            .get_column('result_home')
    )

    predictions_all: dict[str, np.ndarray | pl.Series] = (
        model_predictions | {'elo_baseline': predictions_elo}
    )

    accuracies: dict[str, float] = {
        model_name: float(
            accuracy_score(y_test_results, predictions)
        )
        for model_name, predictions in predictions_all.items()
    }

    accuracy_summary: pl.DataFrame = (
        pl.DataFrame({
            'model': accuracies.keys(),
            'accuracy': accuracies.values()
        })
        .sort(pl.col('model') != 'elo_baseline', 'accuracy', descending=True)
        .with_columns(
            pl.col('accuracy')
                .map_elements(
                    lambda x: f'{x * 100:.2f}%', 
                    return_dtype=pl.Utf8
                )
                .alias('accuracy')
        )
    )

    return accuracy_summary
