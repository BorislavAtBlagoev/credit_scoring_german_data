import pandas as pd
import numpy as np


def logistic_coefficients(model, preprocessor):
    clf = model.named_steps["clf"]
    ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]

    numeric_features = preprocessor.transformers_[0][2]
    cat_features = list(ohe.get_feature_names_out())

    all_features = list(numeric_features) + cat_features

    coefs = clf.coef_[0]
    df = pd.DataFrame({
        "feature": all_features,
        "coef": coefs,
        "abs_coef": np.abs(coefs)
    }).sort_values("abs_coef", ascending=False)

    return df.head(15)


def random_forest_importance(model, preprocessor):
    clf = model.named_steps["clf"]
    ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]

    numeric_features = list(preprocessor.transformers_[0][2])
    cat_features = list(ohe.get_feature_names_out())

    all_features = numeric_features + cat_features
    importances = clf.feature_importances_

    df = pd.DataFrame({
        "feature": all_features,
        "importance": importances
    }).sort_values("importance", ascending=False)

    return df.head(15)
