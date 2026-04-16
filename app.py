import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.linear_model import LinearRegression

# =========================
# CARREGAR DADOS
# =========================
df = pd.read_csv("datatran2021.csv", encoding="latin1", sep=";")
df.columns = df.columns.str.strip()

# =========================
# CONFIG
# =========================
menu = st.sidebar.radio(
    "📌 Navegação",
    [
        "🏠 Dashboard Principal",
        "⚠️ Impacto e Severidade",
        "📈 Intervalos de Confiança",
        "🧪 Testes de Hipóteses",
        "📉 Regressão Linear"
    ]
)
if menu == "🏠 Dashboard Principal":
    st.set_page_config(layout="wide")
    st.title("🚗 Análise de Acidentes Rodoviários")

    sns.set_theme(style="whitegrid")

    # =========================
    # TRATAMENTO
    # =========================
    df["hora"] = pd.to_datetime(df["horario"], errors="coerce").dt.hour

    ordem_dias = [
        "segunda-feira", "terça-feira", "quarta-feira",
        "quinta-feira", "sexta-feira", "sábado", "domingo"
    ]

    # =========================
    # SIDEBAR - FILTRO
    # =========================
    uf = st.sidebar.selectbox(
        "Filtrar por UF",
        ["Todos"] + sorted(df["uf"].dropna().unique())
    )

    if uf != "Todos":
        df = df[df["uf"] == uf]

    # =========================
    # FILTRO TOP CAUSAS
    # =========================
    top_causas = df["causa_acidente"].value_counts().head(10).index
    df_top = df[df["causa_acidente"].isin(top_causas)]

    # =========================
    # KPIs
    # =========================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total de Acidentes", len(df))
    col2.metric("Estados", df["uf"].nunique())
    col3.metric("Causas diferentes", df["causa_acidente"].nunique())
    col4.metric(
        "Com vítimas",
        df["classificacao_acidente"].value_counts().get("Com Vítimas Feridas", 0)
    )

    # =========================
    # INSIGHT AUTOMÁTICO
    # =========================
    st.markdown("### 📌 Insight automático")

    hora_pico = df["hora"].value_counts().idxmax()
    dia_pico = df["dia_semana"].value_counts().idxmax()
    causa_top = df["causa_acidente"].value_counts().idxmax()

    st.info(
        f"""
        🔥 Horário com mais acidentes: {hora_pico}h  
        📅 Dia mais crítico: {dia_pico}  
        ⚠️ Principal causa: {causa_top}
        """
    )

    # =========================
    # LAYOUT DOS GRÁFICOS
    # =========================
    st.subheader("📊 Análise de Padrões")

    colA, colB = st.columns(2)

    # =========================
    # GRÁFICO 1 — CAUSA x HORA
    # =========================
    with colA:
        st.markdown("### Causas por Hora")

        causa_hora = df_top.groupby(["hora", "causa_acidente"]).size().unstack().fillna(0)

        fig1, ax1 = plt.subplots(figsize=(12, 5))
        causa_hora.plot(ax=ax1)

        ax1.set_title("Top 10 causas de acidente por hora")
        ax1.set_xlabel("Hora do dia")
        ax1.set_ylabel("Quantidade de acidentes")
        ax1.set_xticks(range(24))
        ax1.grid(alpha=0.3)
        ax1.legend(title="Causa", bbox_to_anchor=(1.05, 1))

        plt.tight_layout()
        st.pyplot(fig1)

    # =========================
    # GRÁFICO 2 — CAUSA x DIA
    # =========================
    with colB:
        st.markdown("### Causas por Dia da Semana")

        causa_dia = df_top.groupby(["dia_semana", "causa_acidente"]).size().unstack().fillna(0)
        causa_dia = causa_dia.reindex(ordem_dias)

        fig2, ax2 = plt.subplots(figsize=(12, 5))
        causa_dia.plot(ax=ax2)

        ax2.set_title("Top 10 causas de acidente por dia da semana")
        ax2.set_xlabel("Dia da semana")
        ax2.set_ylabel("Quantidade de acidentes")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(alpha=0.3)
        ax2.legend(title="Causa", bbox_to_anchor=(1.05, 1))

        plt.tight_layout()
        st.pyplot(fig2)

    # =========================
    # GRÁFICO 3 — HORA x DIA (DESTAQUE)
    # =========================
    st.markdown("## 🔥 Padrão de Acidentes (Hora x Dia)")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        hora_dia = df.groupby(["hora", "dia_semana"]).size().unstack().fillna(0)
        hora_dia = hora_dia.reindex(columns=ordem_dias)

        fig3, ax3 = plt.subplots(figsize=(9, 4))

        sns.heatmap(hora_dia.T, cmap="Blues", ax=ax3)

        ax3.set_xlabel("Hora do dia")
        ax3.set_ylabel("Dia da semana")
        ax3.set_title("Acidentes por hora e dia da semana")

        st.pyplot(fig3)

elif menu == "⚠️ Impacto e Severidade":

    st.title("⚠️ Impacto e Severidade dos Acidentes")
    
    st.markdown("Análise de gravidade, risco e distribuição dos acidentes.")
    st.divider()

    # =========================
    # CRIAÇÃO DA SEVERIDADE
    # =========================
    df["severidade"] = (
        df["mortos"] * 3 +
        df["feridos_graves"] * 2 +
        df["feridos_leves"] * 1
    )

    # =========================
    # KPIs (CARDS)
    # =========================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total de Acidentes", len(df))
    col2.metric("Total de Mortos", df["mortos"].sum())
    col3.metric("Feridos Graves", df["feridos_graves"].sum())
    col4.metric("Severidade Média", round(df["severidade"].mean(), 2))

    st.divider()

    # =========================
    # VARIÁVEIS BASE
    # =========================
    class_tipo = pd.crosstab(
        df["tipo_acidente"],
        df["classificacao_acidente"],
        normalize="index"
    )

    pessoas_analise = (
        df.groupby("pessoas")[["mortos", "feridos_leves"]]
        .mean()
        .sort_index()
    )

    pista_class = pd.crosstab(
        df["tipo_pista"],
        df["classificacao_acidente"],
        normalize="index"
    )

    prob_morte = df.groupby("tipo_acidente")["mortos"].mean()

    uf_perigo = (
        df.groupby("uf")["severidade"]
        .mean()
        .sort_values(ascending=False)
    )

    # =========================
    # ESTILO
    # =========================
    plt.style.use("seaborn-v0_8-darkgrid")

    # =========================
    # GRÁFICO 1
    # =========================
    fig1, ax1 = plt.subplots(figsize=(9, 4))
    pessoas_analise.plot(kind="bar", ax=ax1)

    ax1.set_title("Impacto médio por quantidade de pessoas")
    ax1.set_xlabel("Número de pessoas")
    ax1.set_ylabel("Média de vítimas")
    ax1.grid(axis="y", alpha=0.3)

    # =========================
    # GRÁFICO 2
    # =========================
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    class_tipo.plot(kind="barh", stacked=True, ax=ax2, colormap="viridis")

    ax2.set_title("Distribuição da gravidade por tipo de acidente")
    ax2.set_xlabel("Proporção")
    ax2.set_ylabel("")
    ax2.legend(title="Classificação", bbox_to_anchor=(1.05, 1))

    # =========================
    # GRÁFICO 3
    # =========================
    fig3, ax3 = plt.subplots(figsize=(9, 4))

    pista_class.plot(
        kind="bar",
        stacked=True,
        ax=ax3,
        colormap="viridis"
    )

    ax3.set_title("Gravidade por tipo de pista")
    ax3.set_xlabel("")
    ax3.set_ylabel("Proporção")
    ax3.tick_params(axis="x", rotation=20)
    ax3.grid(axis="y", alpha=0.3)

    ax3.legend(title="Classificação", bbox_to_anchor=(1.05, 1))
    # =========================
    # GRÁFICO 4
    # =========================
    fig4, ax4 = plt.subplots(figsize=(9, 4))
    prob_morte.plot(kind="barh", ax=ax4)

    ax4.set_title("Probabilidade de morte por tipo de acidente")
    ax4.set_xlabel("Probabilidade")
    ax4.set_ylabel("")

    # =========================
    # GRÁFICO 5
    # =========================
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    uf_perigo.head(15).plot(kind="bar", ax=ax5, color="#E45756")

    ax5.set_title("Top 15 estados com maior severidade média")
    ax5.set_ylabel("Severidade média")
    ax5.set_xlabel("")
    ax5.tick_params(axis="x", rotation=45)

    # =========================
    # EXPANDERS (ORGANIZAÇÃO)
    # =========================
    with st.expander("👥 Impacto por número de pessoas"):
        st.pyplot(fig1)

    with st.expander("📊 Classificação dos acidentes"):
        st.pyplot(fig2)

    with st.expander("🛣️ Tipo de pista"):
        st.pyplot(fig3)

    with st.expander("☠️ Probabilidade de morte"):
        st.pyplot(fig4)

    with st.expander("🗺️ Severidade por estado"):
        st.pyplot(fig5)

if menu == "📈 Intervalos de Confiança":

    df["severidade"] = (
    df["mortos"] * 10 +
    df["feridos_leves"] * 2 +
    df["pessoas"] * 0.5
    )

    st.title("📈 Intervalo de Confiança (95%)")

    st.markdown("""
    Aqui estimamos a faixa onde o valor médio provavelmente está.

    👉 Exemplo:
    - Severidade média de acidentes em cada fase do dia
    """)

    variavel = st.selectbox(
        "📌 Variável",
        ["severidade", "mortos", "feridos_leves", "pessoas"]
    )

    grupo = st.selectbox(
        "📊 Agrupar por",
        ["fase_dia", "tipo_acidente", "uf", "condicao_metereologica"]
    )

    st.divider()

    def ic(x):
        x = x.dropna()
        if len(x) < 2:
            return (np.nan, np.nan)

        return stats.t.interval(
            0.95,
            len(x) - 1,
            loc=np.mean(x),
            scale=stats.sem(x)
        )

    ic_df = df.groupby(grupo)[variavel].apply(ic)

    # transformar em tabela mais legível
    ic_df = ic_df.apply(pd.Series)
    ic_df.columns = ["Limite Inferior", "Limite Superior"]

    st.subheader("📊 Resultado")

    st.dataframe(ic_df.style.format("{:.3f}"))

    st.markdown("""
    ### 📌 Como interpretar:
    - Cada linha = um grupo (ex: “Plena Noite”)
    - Intervalo = faixa onde a média real provavelmente está
    - Quanto menor o intervalo → mais confiável a média
    """)

if menu == "🧪 Testes de Hipóteses":

    df["severidade"] = (
    df["mortos"] * 10 +
    df["feridos_leves"] * 2 +
    df["pessoas"] * 0.5
    )

    st.title("🧪 Teste de Diferenças entre Grupos (ANOVA)")

    st.markdown("""
    Aqui verificamos se existe **diferença real** entre médias de um grupo.

    Exemplo:
    - Os acidentes são mais graves à noite do que de dia?
    - Alguns tipos de pista são mais perigosos?
    """)

    variavel = st.selectbox(
        "📌 O que você quer analisar?",
        ["severidade", "mortos", "feridos_leves", "pessoas"]
    )

    grupo = st.selectbox(
        "📊 Comparar por:",
        ["fase_dia", "tipo_acidente", "uf", "condicao_metereologica", "tipo_pista"]
    )

    st.divider()

    # -----------------------------
    # ANOVA
    # -----------------------------
    grupos = [
        df[df[grupo] == g][variavel].dropna()
        for g in df[grupo].dropna().unique()
    ]

    f_stat, p_val = stats.f_oneway(*grupos)

    st.subheader("📊 Resultado do Teste ANOVA")

    col1, col2 = st.columns(2)
    col1.metric("F-statistic", f"{f_stat:.3f}")
    col2.metric("P-value", f"{p_val:.5f}")

    st.divider()

    # -----------------------------
    # INTERPRETAÇÃO SIMPLES
    # -----------------------------
    if p_val < 0.05:
        st.error("🚨 Existe diferença estatisticamente significativa entre os grupos.")

        st.markdown("""
        👉 Isso significa que **pelo menos um grupo é diferente dos outros**.
        Mas ainda não sabemos qual.
        """)

        tukey = pairwise_tukeyhsd(
            endog=df[variavel],
            groups=df[grupo],
            alpha=0.05
        )

        st.subheader("🔬 Comparações detalhadas (Tukey HSD)")
        st.text(str(tukey))

        st.info("""
        📌 O teste de Tukey mostra **quais grupos são diferentes entre si**.
        """)

    else:
        st.success("✅ Não há diferença significativa entre os grupos.")

        st.markdown("""
        👉 Isso sugere que os grupos têm médias **estatisticamente semelhantes**.
        """)

    st.divider()

    # -----------------------------
    # EXPLICAÇÃO FINAL
    # -----------------------------
    st.caption("""
    ℹ️ ANOVA testa se existe diferença geral entre grupos.
    Se der significativo, usamos Tukey para descobrir onde está a diferença.
    """)

if menu == "📉 Regressão Linear":

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error

    st.title("📉 Regressão Linear - Fatores de Severidade")

    st.markdown("""
    Aqui o modelo tenta entender como:
    - tipo de acidente  
    - dia da semana  
    - horário  

    influenciam a gravidade dos acidentes.
    """)

    # severidade
    df["severidade"] = (
        df["mortos"] * 10 +
        df["feridos_leves"] * 2 +
        df["pessoas"] * 0.5
    )

    df["hora"] = pd.to_datetime(df["horario"], errors="coerce").dt.hour

    X = df[["tipo_acidente", "dia_semana", "hora"]]
    y = df["severidade"]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    col1, col2 = st.columns(2)
    col1.metric("📊 R² do modelo", round(r2, 3))
    col2.metric("📉 Erro (MSE)", round(mse, 2))

    st.subheader("📌 Influência das variáveis")

    coef_df = pd.DataFrame({
        "Variável": X.columns,
        "Impacto no risco": model.coef_
    }).sort_values("Impacto no risco", ascending=False)

    st.dataframe(coef_df)

    st.subheader("🎯 Real vs Previsto")

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Real")
    ax.set_ylabel("Previsto")
    ax.set_title("Comparação")

    st.pyplot(fig)