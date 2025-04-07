from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from modeling import get_classifiers

import config


def run_grid_search(X, y, classifier_name="RandomForest"):
    # Create a pipeline template for one classifier (e.g., RandomForest)
    classifier = get_classifiers()[classifier_name]

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('pca', PCA(random_state=config.RANDOM_STATE)),  # n_components to be tuned
        ('model', classifier)
    ])

    param_grid = {
        'pca__n_components': [0.90, 0.95, 0.99]  # or fixed numbers: [30, 50, 70]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted')
    grid_search.fit(X, y)
    #print("Best PCA setting:", grid_search.best_params_)
    return grid_search.best_params_, grid_search.best_score_
