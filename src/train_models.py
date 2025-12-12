from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score


def train_logreg(preprocessor):
    clf = LogisticRegression(max_iter=30000)
    model = Pipeline([
        ("prep", preprocessor),
        ("clf", clf)
    ])
    return model


def train_rf(preprocessor):
    rf = RandomForestClassifier(
        n_estimators=200, random_state=42
    )
    model = Pipeline([
        ("prep", preprocessor),
        ("clf", rf)
    ])
    return model


def evaluate(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"==== {name} ====")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
