import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, 
    OneHotEncoder, 
    PolynomialFeatures
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PoissonRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV


def select_model(matches: pd.DataFrame) -> Pipeline:
    X_train, X_test, y_train, y_test = split_data(matches)
    
    model: Pipeline = train_model(X_train, y_train)

    return model


def split_data(matches: pd.DataFrame) -> list[pd.DataFrame]:
    features: list[str] = (
        ['tournament']
            + [
                f'{var}_{team_type}' 
                for var in [
                    'elo',
                    'form_avg_goals_for', 
                    'form_avg_goals_against',
                    'form_avg_win',
                    'form_avg_loss'
                ]
                for team_type in ('home', 'away')
            ]
            + [
                'h2h_rate_win_home',
                'h2h_rate_loss_home',
                'neutral'
            ]
    )

    labels: list[str] = ['goals_for_home', 'goals_for_away']

    X: pd.DataFrame = matches[features]

    y: pd.DataFrame = matches[labels]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    return [X_train, X_test, y_train, y_test]


def train_model(
        X_train: pd.DataFrame, 
        y_train: pd.DataFrame
) -> Pipeline:     
    column_transformer: ColumnTransformer = ColumnTransformer(
        [
            (
                'categorical', 
                OneHotEncoder(), 
                X_train.columns[X_train.dtypes == 'object']
            ),
            (
                'numeric', 
                StandardScaler(), 
                X_train.columns[X_train.dtypes != 'object']
            )
        ]
    )

    pipeline: Pipeline = Pipeline(
        [
            ('tfrm_cols', column_transformer),
            ('tfrm_poly', PolynomialFeatures()),
            ('model', MultiOutputRegressor(PoissonRegressor()))
        ]
    )

    param_grid: dict[str, list[int]] = {
        'tfrm_poly__degree': [1, 2, 3]
    }

    grid_search: GridSearchCV = GridSearchCV(pipeline, param_grid, cv=5)

    grid_search.fit(X_train, y_train)

    model: Pipeline = grid_search.best_estimator_

    return model