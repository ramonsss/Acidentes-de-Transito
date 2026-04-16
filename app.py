import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        "📊 Análises Extras"
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