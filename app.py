import os
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


# =========================
# Configurações gerais
# =========================
TARGET_COL = "Obesity"

ORDINAL_UI = {
    "FCVC": [1, 2, 3],
    "NCP": [1, 2, 3, 4],
    "CH2O": [1, 2, 3],
    "FAF": [0, 1, 2, 3],
    "TUE": [0, 1, 2],
}

ORDINAL_BOUNDS = {
    "FCVC": (1, 3),
    "NCP": (1, 4),
    "CH2O": (1, 3),
    "FAF": (0, 3),
    "TUE": (0, 2),
}

ORDINAL_ROUND_COLS = list(ORDINAL_BOUNDS.keys())


# =========================
# Utilidades
# =========================
def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ORDINAL_ROUND_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")
            lo, hi = ORDINAL_BOUNDS[col]
            df[col] = df[col].clip(lo, hi)
    return df


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv("Obesity.csv")


def train_and_save_model(
    df: pd.DataFrame,
    model_path: str = "model.joblib",
    metrics_path: str = "metrics.json",
) -> Pipeline:

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
        ]
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

    joblib.dump(pipeline, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": acc,
                "classification_report": report,
                "notes": {
                    "ordinal_rounded_cols": ORDINAL_ROUND_COLS,
                    "ordinal_bounds": ORDINAL_BOUNDS,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return pipeline


@st.cache_resource
def load_model(df: pd.DataFrame) -> Pipeline:
    if not os.path.exists("model.joblib"):
        st.info("Modelo não encontrado. Treinando automaticamente (primeira execução)...")
        return train_and_save_model(df)
    return joblib.load("model.joblib")


# =========================
# Páginas
# =========================
def prediction_page(df: pd.DataFrame):
    st.title("Predição do Nível de Obesidade")

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    input_data = {}

    col1, col2 = st.columns(2)

    for i, col in enumerate(feature_cols):
        container = col1 if i % 2 == 0 else col2

        if col in ORDINAL_UI:
            input_data[col] = container.selectbox(col, ORDINAL_UI[col])
        elif df[col].dtype == "object":
            options = sorted(df[col].dropna().unique())
            input_data[col] = container.selectbox(col, options)
        else:
            series = pd.to_numeric(df[col], errors="coerce")
            input_data[col] = container.number_input(
                col,
                min_value=float(series.min()),
                max_value=float(series.max()),
                value=float(series.median()),
            )

    if st.button("Prever"):
        model = load_model(df)
        X = pd.DataFrame([input_data])
        pred = model.predict(X)[0]
        st.success(f"✅ Nível de obesidade previsto: **{pred}**")

        if hasattr(model, "predict_proba"):
            proba_df = pd.DataFrame(
                {
                    "Classe": model.classes_,
                    "Probabilidade": model.predict_proba(X)[0],
                }
            ).sort_values("Probabilidade", ascending=False)
            st.subheader("Probabilidades por classe")
            st.dataframe(proba_df, use_container_width=True)


def analytics_page(df: pd.DataFrame):
    st.title("Painel Analítico — Insights sobre Obesidade")

    st.subheader("Distribuição das classes")
    st.bar_chart(df[TARGET_COL].value_counts())

    st.subheader("IMC (BMI) por classe")
    df_bmi = df.copy()
    df_bmi["BMI"] = df_bmi["Weight"] / (df_bmi["Height"] ** 2)

    fig, ax = plt.subplots()
    groups, labels = [], []

    for cls in df_bmi[TARGET_COL].unique():
        groups.append(df_bmi[df_bmi[TARGET_COL] == cls]["BMI"].dropna())
        labels.append(cls)

    ax.boxplot(groups, labels=labels)
    plt.xticks(rotation=45, ha="right")
    ax.set_ylabel("BMI (kg/m²)")
    st.pyplot(fig)

    if os.path.exists("metrics.json"):
        with open("metrics.json", "r", encoding="utf-8") as f:
            metrics = json.load(f)
        st.metric("Acurácia do modelo", f"{metrics['accuracy']:.4f}")


# =========================
# Main
# =========================
def main():
    st.set_page_config(page_title="Diagnóstico de Obesidade", layout="wide")

    df = load_data()

    st.sidebar.title("Navegação")
    page = st.sidebar.radio("Ir para:", ["Predição", "Painel Analítico"])

    if page == "Predição":
        prediction_page(df)
    else:
        analytics_page(df)


if __name__ == "__main__":
    main()