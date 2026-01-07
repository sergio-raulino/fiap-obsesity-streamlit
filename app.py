import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


TARGET_COL = "Obesity"

# Ordinais (UI com categorias válidas)
ORDINAL_UI = {
    "FCVC": [1, 2, 3],
    "NCP": [1, 2, 3, 4],
    "CH2O": [1, 2, 3],
    "FAF": [0, 1, 2, 3],
    "TUE": [0, 1, 2],
}


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


@st.cache_data
def load_data():
    return pd.read_csv("Obesity.csv")


def prediction_page(df: pd.DataFrame):
    st.title("Predição do Nível de Obesidade")

    if TARGET_COL not in df.columns:
        st.error(f"Coluna alvo '{TARGET_COL}' não encontrada no CSV.")
        st.stop()

    st.write(
        "Preencha os dados do paciente e clique em **Prever**. "
        "O modelo retornará a classe prevista (nível de obesidade)."
    )

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    input_data = {}

    col1, col2 = st.columns(2)

    for i, col in enumerate(feature_cols):
        container = col1 if i % 2 == 0 else col2

        # Ordinais (categorias)
        if col in ORDINAL_UI:
            input_data[col] = container.selectbox(col, ORDINAL_UI[col], index=0)
            continue

        # Categóricos
        if df[col].dtype == "object":
            options = sorted(df[col].dropna().unique().tolist())
            input_data[col] = container.selectbox(col, options)
            continue

        # Numéricos contínuos
        series = pd.to_numeric(df[col], errors="coerce")
        vmin = float(series.min())
        vmax = float(series.max())
        default = float(series.median())
        input_data[col] = container.number_input(
            col, min_value=vmin, max_value=vmax, value=default
        )

    if st.button("Prever"):
        model = load_model()
        X = pd.DataFrame([input_data])

        pred = model.predict(X)[0]
        st.success(f"✅ Nível de obesidade previsto: **{pred}**")

        # Probabilidades (se suportado)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            classes = model.classes_
            proba_df = (
                pd.DataFrame({"classe": classes, "probabilidade": proba})
                .sort_values("probabilidade", ascending=False)
                .reset_index(drop=True)
            )
            st.subheader("Probabilidades por classe")
            st.dataframe(proba_df, use_container_width=True)


def analytics_page(df: pd.DataFrame):
    st.title("Painel Analítico — Insights sobre Obesidade")

    if TARGET_COL not in df.columns:
        st.error(f"Coluna alvo '{TARGET_COL}' não encontrada no CSV.")
        st.stop()

    # Distribuição das classes
    st.subheader("Distribuição das classes (alvo)")
    st.bar_chart(df[TARGET_COL].value_counts())

    # BMI (IMC)
    st.subheader("IMC (BMI) por classe")
    if "Weight" in df.columns and "Height" in df.columns:
        df_bmi = df.copy()
        df_bmi["BMI"] = df_bmi["Weight"] / (df_bmi["Height"] ** 2)

        fig, ax = plt.subplots()
        groups, labels = [], []

        for cls in df_bmi[TARGET_COL].dropna().unique():
            groups.append(df_bmi.loc[df_bmi[TARGET_COL] == cls, "BMI"].dropna().values)
            labels.append(str(cls))

        ax.boxplot(groups, labels=labels)
        ax.set_xlabel("Classe (Obesity)")
        ax.set_ylabel("BMI (kg/m²)")
        ax.set_title("Distribuição de BMI por classe")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig, clear_figure=True)
    else:
        st.warning("Colunas 'Weight' e 'Height' não encontradas; não foi possível calcular BMI.")

    # Métricas do modelo
    st.subheader("Métricas do modelo")
    try:
        with open("metrics.json", "r", encoding="utf-8") as f:
            metrics = json.load(f)

        st.metric("Acurácia (teste)", f"{metrics.get('accuracy', 0):.4f}")

        report = metrics.get("classification_report", {})
        summary = {k: report.get(k, {}) for k in ["macro avg", "weighted avg"] if k in report}
        st.write("Resumo do relatório:")
        st.json(summary)

        notes = metrics.get("notes", {})
        if notes:
            st.caption("Notas da pipeline:")
            st.json(notes)

    except FileNotFoundError:
        st.error("Não encontrei metrics.json. Execute primeiro: `python train_model.py`")


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
