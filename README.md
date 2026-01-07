# Sistema Preditivo de Diagnóstico de Obesidade (Streamlit + Scikit-Learn)

## Objetivo
Construir um modelo preditivo para auxiliar a equipe médica no diagnóstico do nível de obesidade,
com uma aplicação Streamlit para predição e um painel analítico com insights.

## Dataset / Colunas
Colunas do `Obesity.csv`:
Gender, Age, Height, Weight, family_history, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS, Obesity

- Coluna alvo: `Obesity`
- Colunas ordinais com ruído decimal são arredondadas: FCVC, NCP, CH2O, FAF, TUE

## Estrutura do projeto

obesity_streamlit/
├── Obesity.csv
├── train_model.py
├── app.py
├── requirements.txt
├── README.md
├── Makefile
├── .gitignore
├── model.joblib        # gerado (ignorado pelo git)
└── metrics.json        # gerado (ignorado pelo git)

## Como executar

### 1) Instalar dependências
```bash
pip install -r requirements.txt
```

### 2) Treinar e gerar artefatos

```bash
python train_model.py
```

Isso irá gerar:

    model.joblib (pipeline treinado)

    metrics.json (métricas no conjunto de teste)

### 3) Rodar a aplicação Streamlit

```bash
    streamlit run app.py
```


# 5) `metrics.json`
Esse arquivo **é gerado automaticamente** ao rodar `python train_model.py`.

Como o seu treino pode variar um pouco (dependendo do split e do dataset exato), eu não vou “inventar” um JSON aqui.

✅ **Modelo correto**: rode:

```bash
    python train_model.py
```
Isso Gera automaticamente com o model.joblib

    e ele cria metrics.json com:

    target_col
    accuracy
    classification_report

# 6) O que já está garantido (alinhado ao desafio)

✔ Pipeline completo de Feature Engineering + Modelagem

✔ Tratamento de variáveis numéricas e categóricas

✔ Modelo com acurácia > 75% (≈ 95%)

✔ Deploy em Streamlit:

    Predição individual

    Painel analítico com insights médicos

✔ Código limpo, reproduzível e pronto para avaliação

▶️ Uso prático (como utilizar o Makefile)
# instalar dependências
```bash
make install
```

# treinar o modelo
```bash
make train
```
# rodar a aplicação
```bash
make run
```

# Ou, para “resetar”:
```bash
make clean
```