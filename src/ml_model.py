import xgboost as xgb
import lightgbm as lgb

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def train_ml_model(X_train, y_train):

    cat_cols = X_train.select_dtypes(include="object").columns
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42
        
    )

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=40,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.9,
        verbose=-1,
        random_state=42
    )

    xgb_pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", xgb_model)
    ])

    lgb_pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", lgb_model)
    ])

    xgb_pipeline.fit(X_train, y_train)
    lgb_pipeline.fit(X_train, y_train)

    return xgb_pipeline, lgb_pipeline