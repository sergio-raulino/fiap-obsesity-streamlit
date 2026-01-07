import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


TARGET_COL = "Obesity"

# Colunas ordinais com ruído decimal no arquivo (arredondar para inteiro)
ORDINAL_ROUND_COLS = ["FCVC", "NCP", "CH2O", "FAF", "TUE"]

# Domínios esperados (após arredondar, garantir limites válidos)
ORDINAL_BOUNDS = {
    "FCVC": (1, 3),  # 1..3
    "NCP": (1, 4),   # 1..4
    "CH2O": (1, 3),  # 1..3
    "FAF": (0, 3),   # 0..3
    "TUE": (0, 2),   # 0..2
}


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica limpeza mínima:
    - Converte colunas ordinais para numérico, arredonda e aplica clip no domínio.
    """
    df = df.copy()

    for col in ORDINAL_ROUND_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")
            lo, hi = ORDINAL_BOUNDS[col]
            df[col] = df[col].clip(lower=lo, upper=hi)

    return df


def main() -> None:
    df = pd.read_csv("Obesity.csv")

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Coluna alvo '{TARGET_COL}' não encontrada. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    df = preprocess_df(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    num_cols = X.select_dtypes(include=["int64", "float64", "Int64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
    )

    model = GradientBoostingClassifier(random_state=42)

    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    joblib.dump(pipeline, "model.joblib")

    metrics = {
        "target_col": TARGET_COL,
        "accuracy": acc,
        "classification_report": report,
        "notes": {
            "ordinal_rounded_cols": ORDINAL_ROUND_COLS,
            "ordinal_bounds": ORDINAL_BOUNDS,
        },
    }

    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("✅ Treino concluído")
    print(f"✅ Acurácia (teste): {acc:.4f}")
    print("✅ Gerados: model.joblib, metrics.json")


if __name__ == "__main__":
    main()
