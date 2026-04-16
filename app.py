import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.linear_model import LinearRegression
import folium
from streamlit_folium import st_folium

# =========================
# CARREGAR DADOS
# =========================
df_bruto = pd.read_csv("datatran2021.csv", encoding="latin1", sep=";")


def preparar_dados(base):
    base = base.copy()
    base.columns = base.columns.str.strip()

    texto_limpo = ["dia_semana", "horario", "uf", "municipio", "causa_acidente", "tipo_acidente", "classificacao_acidente", "fase_dia", "sentido_via", "condicao_metereologica", "tipo_pista", "tracado_via", "uso_solo", "regional", "delegacia", "uop"]
    for coluna in texto_limpo:
        if coluna in base.columns:
            base[coluna] = base[coluna].astype(str).str.strip()

    for coluna in ["data_inversa"]:
        if coluna in base.columns:
            base[coluna] = pd.to_datetime(base[coluna], errors="coerce")

    for coluna in ["km", "latitude", "longitude"]:
        if coluna in base.columns:
            base[coluna] = pd.to_numeric(
                base[coluna].astype(str).str.replace(",", ".", regex=False),
                errors="coerce"
            )

    for coluna in ["br", "pessoas", "mortos", "feridos_leves", "feridos_graves", "ilesos", "ignorados", "feridos", "veiculos"]:
        if coluna in base.columns:
            base[coluna] = pd.to_numeric(base[coluna], errors="coerce")

    if "br" in base.columns:
        base["br"] = base["br"].round().astype("Int64")

    if "horario" in base.columns:
        base["hora"] = pd.to_datetime(base["horario"], format="%H:%M:%S", errors="coerce").dt.hour

    if "data_inversa" in base.columns:
        base["mes"] = base["data_inversa"].dt.month

    return base


def criar_severidade(base):
    return (
        base["mortos"].fillna(0) * 10 +
        base["feridos_leves"].fillna(0) * 2 +
        base["pessoas"].fillna(0) * 0.5
    )


df = preparar_dados(df_bruto)

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
        "📉 Regressão Linear",
        "🧼 Dados Limpos e Preparados",
        "📦 Box Plot e Outliers",
        "🔍 Análise Detalhada",
        "🧩 Perfil Operacional da Via"
    ]
)
if menu == "🏠 Dashboard Principal":
    st.set_page_config(layout="wide")
    st.title("🚗 Análise de Acidentes Rodoviários")
    st.caption("Painel executivo para identificar onde, quando e por que os acidentes se concentram.")

    with st.expander("🧭 Como ler este painel"):
        st.markdown("""
        - Use o filtro de UF para olhar padrões locais ou nacionais.
        - Observe o horário e o dia com maior incidência para priorizar ações operacionais.
        - Compare as linhas dos gráficos para perceber quais causas mudam ao longo do dia.
        - A matriz Hora x Dia ajuda a identificar picos simultâneos de risco.
        """)

    sns.set_theme(style="whitegrid")

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
    acidentes_com_vitima = (df["mortos"].fillna(0) + df["feridos"].fillna(0) > 0).sum()
    perc_com_vitima = (acidentes_com_vitima / len(df) * 100) if len(df) else 0
    br_mais_critica = df["br"].value_counts().idxmax() if df["br"].notna().any() else "N/A"

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total de Acidentes", len(df))
    col2.metric("Estados", df["uf"].nunique())
    col3.metric("Causas diferentes", df["causa_acidente"].nunique())
    col4.metric("Acidentes com vítimas", f"{acidentes_com_vitima} ({perc_com_vitima:.1f}%)")

    st.caption(f"Rodovia com maior volume de registros: BR-{br_mais_critica}")

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

    st.markdown("### 🚧 Rodovias com maior concentração")
    top_brs = (
        df.dropna(subset=["br"]) 
        .groupby("br")
        .size()
        .reset_index(name="Total de acidentes")
        .sort_values("Total de acidentes", ascending=False)
        .head(10)
    )
    if not top_brs.empty:
        st.dataframe(top_brs.rename(columns={"br": "BR"}), width="stretch")

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
    st.caption("Áreas mais escuras na matriz indicam combinação de hora e dia com maior volume relativo de acidentes.")


elif menu == "⚠️ Impacto e Severidade":

    st.title("⚠️ Impacto e Severidade dos Acidentes")

    st.markdown("Análise de gravidade, risco e distribuição dos acidentes.")
    st.info(
        "Severidade aqui é um índice ponderado de impacto: mortos têm peso maior que feridos graves, e feridos graves têm peso maior que feridos leves."
    )
    with st.expander("📘 O que significa a severidade nesta seção"):
        st.markdown("""
        Fórmula usada nesta página:
        **severidade = mortos × 3 + feridos_graves × 2 + feridos_leves × 1**

        - Não é probabilidade.
        - Não representa risco individual.
        - É um indicador comparativo para priorizar grupos, estados e tipos de acidente.
        """)
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
        st.caption("Mostra como a média de vítimas muda conforme aumenta o número de pessoas envolvidas.")
        st.pyplot(fig1)

    with st.expander("📊 Classificação dos acidentes"):
        st.caption("Cada barra soma 100% dentro do tipo de acidente e mostra a proporção de classificações.")
        st.pyplot(fig2)

    with st.expander("🛣️ Tipo de pista"):
        st.caption("Compara a distribuição de gravidade entre pista simples, dupla e outros tipos de via.")
        st.pyplot(fig3)

    with st.expander("☠️ Probabilidade de morte"):
        st.caption("Leitura por média de mortos por tipo de acidente; valores maiores indicam tipos mais letais.")
        st.pyplot(fig4)

    with st.expander("🗺️ Severidade por estado"):
        st.caption("Ranking dos estados com maior severidade média para apoiar priorização territorial.")
        st.pyplot(fig5)

if menu == "📈 Intervalos de Confiança":

    df_ic = df.copy()
    df_ic["severidade"] = criar_severidade(df_ic)

    st.title("📈 Intervalo de Confiança (95%)")

    st.markdown("""
    Aqui estimamos a faixa onde o valor médio provavelmente está.

    👉 Exemplo:
    - Severidade média de acidentes em cada fase do dia
    """)

    variaveis_numericas = [
        coluna
        for coluna in [
            "severidade", "mortos", "feridos_leves", "feridos_graves", "feridos",
            "pessoas", "veiculos", "ilesos", "ignorados", "km"
        ]
        if coluna in df_ic.columns
    ]

    variavel = st.selectbox("📌 Variável", variaveis_numericas)

    grupos_disponiveis = [
        coluna
        for coluna in [
            "fase_dia", "tipo_acidente", "uf", "condicao_metereologica", "tipo_pista",
            "tracado_via", "sentido_via", "uso_solo", "br", "municipio", "causa_acidente"
        ]
        if coluna in df_ic.columns
    ]

    grupo = st.selectbox("📊 Agrupar por", grupos_disponiveis)

    grupos_muito_grandes = {"municipio", "br", "causa_acidente", "tipo_acidente"}
    top_n = None
    if grupo in grupos_muito_grandes:
        top_n = st.slider(
            "Limitar análise aos grupos mais frequentes",
            min_value=10,
            max_value=80,
            value=25,
            step=5,
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

    base_ci = df_ic[[grupo, variavel]].dropna().copy()
    if top_n is not None:
        categorias_top = base_ci[grupo].value_counts().head(top_n).index
        base_ci = base_ci[base_ci[grupo].isin(categorias_top)]

    resumo_ci = base_ci.groupby(grupo).agg(
        qtd=(variavel, "count"),
        media=(variavel, "mean"),
        desvio=(variavel, "std")
    ).reset_index()

    intervalos = base_ci.groupby(grupo)[variavel].apply(ic).apply(pd.Series)
    intervalos.columns = ["Limite Inferior", "Limite Superior"]
    ic_df = resumo_ci.merge(intervalos, left_on=grupo, right_index=True, how="left")
    ic_df["Amplitude"] = ic_df["Limite Superior"] - ic_df["Limite Inferior"]
    ic_df = ic_df.sort_values("qtd", ascending=False)

    st.subheader("📊 Resultado")

    st.dataframe(
        ic_df.rename(columns={grupo: "Grupo", "qtd": "N", "media": "Média", "desvio": "Desvio Padrão"})
        .style.format({
            "Média": "{:.3f}",
            "Desvio Padrão": "{:.3f}",
            "Limite Inferior": "{:.3f}",
            "Limite Superior": "{:.3f}",
            "Amplitude": "{:.3f}",
        }),
        use_container_width=True
    )

    grupos_pequenos = ic_df[ic_df["qtd"] < 30]
    if not grupos_pequenos.empty:
        st.warning(
            f"{len(grupos_pequenos)} grupos têm menos de 30 observações. Os intervalos nesses casos ficam menos estáveis."
        )

    top_ic = ic_df.head(10).copy()
    fig_ic, ax_ic = plt.subplots(figsize=(10, 5))
    ax_ic.errorbar(
        top_ic["media"],
        top_ic[grupo].astype(str),
        xerr=[
            top_ic["media"] - top_ic["Limite Inferior"],
            top_ic["Limite Superior"] - top_ic["media"]
        ],
        fmt="o",
        color="#4C78A8",
        ecolor="#4C78A8",
        capsize=4,
    )
    ax_ic.set_title(f"Intervalo de confiança por {grupo}")
    ax_ic.set_xlabel(variavel)
    ax_ic.set_ylabel(grupo)
    ax_ic.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_ic)

    st.markdown("""
    ### 📌 Como interpretar:
    - Cada linha = um grupo (ex: “Plena Noite”)
    - O intervalo representa a incerteza da média do grupo, não a dispersão individual dos dados
    - Quanto menor o intervalo, mais estável tende a ser a estimativa da média
    """)

if menu == "🧪 Testes de Hipóteses":

    df_test = df.copy()
    df_test["severidade"] = criar_severidade(df_test)

    st.title("🧪 Teste de Diferenças entre Grupos (ANOVA)")

    st.markdown("""
    Aqui verificamos se existe **diferença real** entre médias de um grupo.

    Exemplo:
    - Os acidentes são mais graves à noite do que de dia?
    - Alguns tipos de pista são mais perigosos?
    """)

    variaveis_numericas = [
        coluna
        for coluna in [
            "severidade", "mortos", "feridos_leves", "feridos_graves", "feridos",
            "pessoas", "veiculos", "ilesos", "ignorados", "km"
        ]
        if coluna in df_test.columns
    ]

    variavel = st.selectbox("📌 O que você quer analisar?", variaveis_numericas)

    grupos_disponiveis = [
        coluna
        for coluna in [
            "fase_dia", "tipo_acidente", "uf", "condicao_metereologica", "tipo_pista",
            "tracado_via", "sentido_via", "uso_solo", "br", "municipio", "causa_acidente"
        ]
        if coluna in df_test.columns
    ]

    grupo = st.selectbox("📊 Comparar por:", grupos_disponiveis)

    grupos_muito_grandes = {"municipio", "br", "causa_acidente", "tipo_acidente"}
    top_n = None
    if grupo in grupos_muito_grandes:
        top_n = st.slider(
            "Limitar ANOVA aos grupos mais frequentes",
            min_value=10,
            max_value=80,
            value=25,
            step=5,
        )

    st.divider()

    base_teste = df_test[[grupo, variavel]].dropna().copy()
    if top_n is not None:
        categorias_top = base_teste[grupo].value_counts().head(top_n).index
        base_teste = base_teste[base_teste[grupo].isin(categorias_top)]

    agrupamentos = [serie[variavel] for _, serie in base_teste.groupby(grupo) if len(serie) >= 2]

    if len(agrupamentos) < 2:
        st.warning("Não há grupos suficientes com pelo menos 2 observações para rodar a ANOVA.")
        st.stop()

    f_stat, p_val = stats.f_oneway(*agrupamentos)
    levene_stat, levene_p = stats.levene(*agrupamentos, center="median")

    media_geral = base_teste[variavel].mean()
    ss_between = sum(len(g) * (g.mean() - media_geral) ** 2 for g in agrupamentos)
    ss_total = ((base_teste[variavel] - media_geral) ** 2).sum()
    eta_quadrado = ss_between / ss_total if ss_total else np.nan

    st.subheader("📊 Resultado do Teste ANOVA")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("F-statistic", f"{f_stat:.3f}")
    col2.metric("P-value", f"{p_val:.5f}")
    col3.metric("Levene p-value", f"{levene_p:.5f}")
    col4.metric("Eta²", f"{eta_quadrado:.3f}")

    resumo_grupos = base_teste.groupby(grupo).agg(
        N=(variavel, "count"),
        Média=(variavel, "mean"),
        Desvio=(variavel, "std")
    ).sort_values("Média", ascending=False)

    st.subheader("📌 Resumo por grupo")
    st.dataframe(resumo_grupos.style.format({"Média": "{:.3f}", "Desvio": "{:.3f}"}), use_container_width=True)

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

        if levene_p < 0.05:
            st.warning(
                "O teste de Levene indica variâncias diferentes entre os grupos. A ANOVA continua informativa, mas a premissa de homogeneidade fica enfraquecida."
            )

        tukey = pairwise_tukeyhsd(
            endog=base_teste[variavel],
            groups=base_teste[grupo],
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
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    st.title("📉 Regressão Linear - Fatores de Severidade")

    st.markdown("""
    Aqui o modelo tenta entender como:
    - tipo de acidente  
    - dia da semana  
    - horário  

    influenciam a gravidade dos acidentes.
    """)

    df_reg = df.copy()
    df_reg["severidade"] = criar_severidade(df_reg)

    alvo = st.selectbox(
        "🎯 Variável alvo",
        ["severidade", "mortos", "feridos_leves", "feridos_graves", "feridos", "pessoas", "veiculos", "km"],
        index=0
    )

    categoricas_disponiveis = [
        coluna
        for coluna in [
            "tipo_acidente", "dia_semana", "uf", "fase_dia", "condicao_metereologica",
            "tipo_pista", "tracado_via", "sentido_via", "uso_solo", "causa_acidente"
        ]
        if coluna in df_reg.columns
    ]

    numericas_disponiveis = [
        coluna
        for coluna in ["hora", "br", "km", "ilesos", "ignorados"]
        if coluna in df_reg.columns and coluna != alvo
    ]

    cat_default = [c for c in ["tipo_acidente", "dia_semana", "fase_dia"] if c in categoricas_disponiveis]
    num_default = [c for c in ["hora", "km", "br"] if c in numericas_disponiveis]

    categoricas_escolhidas = st.multiselect(
        "📚 Preditores categóricos",
        categoricas_disponiveis,
        default=cat_default
    )

    numericas_escolhidas = st.multiselect(
        "🔢 Preditores numéricos",
        numericas_disponiveis,
        default=num_default
    )

    preditores = categoricas_escolhidas + numericas_escolhidas
    if not preditores:
        st.warning("Selecione pelo menos um preditor para treinar a regressão.")
        st.stop()

    colunas_modelo = preditores + [alvo]
    df_reg = df_reg[colunas_modelo].dropna().copy()

    if df_reg.empty:
        st.warning("Não há dados suficientes para treinar a regressão linear.")
        st.stop()

    X = df_reg[preditores]
    y = df_reg[alvo]

    X = pd.get_dummies(X, columns=categoricas_escolhidas, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    residuos = y_test - y_pred

    # métricas
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 R² do modelo", round(r2, 3))
    col2.metric("📉 MSE", round(mse, 2))
    col3.metric("📏 RMSE", round(rmse, 2))
    col4.metric("📐 MAE", round(mae, 2))

    st.subheader("📌 Influência das variáveis")

    coef_df = pd.DataFrame({
        "Variável": X.columns,
        "Impacto no risco": model.coef_
    })
    coef_df["Impacto absoluto"] = coef_df["Impacto no risco"].abs()
    coef_df = coef_df.sort_values("Impacto absoluto", ascending=False)

    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.caption("Maiores efeitos positivos")
        st.dataframe(coef_df.sort_values("Impacto no risco", ascending=False).head(10), use_container_width=True)
    with col_neg:
        st.caption("Maiores efeitos negativos")
        st.dataframe(coef_df.sort_values("Impacto no risco", ascending=True).head(10), use_container_width=True)

    st.caption(f"Preditores usados: {', '.join(preditores)}")
    st.caption("Os coeficientes indicam associação no modelo, não causalidade.")

    st.subheader("🎯 Real vs Previsto")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # pontos com transparência
    ax1.scatter(y_test, y_pred, alpha=0.5, color="#4C78A8")

    # linha ideal (y = x)
    ax1.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle='--'
    )

    ax1.set_xlabel("Real")
    ax1.set_ylabel("Previsto")
    ax1.set_title("Real vs previsto")

    ax2.scatter(y_pred, residuos, alpha=0.5, color="#E45756")
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_xlabel("Previsto")
    ax2.set_ylabel("Resíduo")
    ax2.set_title("Resíduos do modelo")
    ax2.grid(alpha=0.3)

    st.pyplot(fig)

    st.subheader("🧠 Interpretação do modelo")
    st.markdown(f"""
    - O modelo explica aproximadamente **{r2:.1%}** da variação de **{alvo}**.
    - O erro médio absoluto ficou em **{mae:.2f}** unidades de **{alvo}**.
    - Se os resíduos estiverem espalhados sem padrão, a regressão está mais consistente.
    - Se os pontos se afastarem muito da linha 45°, o modelo está perdendo explicação em casos extremos.
    """)

if menu == "🧼 Dados Limpos e Preparados":

    st.title("🧼 Dados Limpos e Preparados")

    st.markdown("""
    Esta sessão documenta o tratamento aplicado na base para deixar o dado pronto para análise.
    O objetivo é reduzir ruído, padronizar tipos e deixar o conjunto consistente para leitura estatística e visual.
    """)

    total_bruto = len(df_bruto)
    total_limpo = len(df)
    duplicadas_brutas = df_bruto.duplicated().sum()
    colunas_convertidas = ["data_inversa", "hora", "km", "latitude", "longitude", "br"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Linhas brutas", f"{total_bruto:,}".replace(",", "."))
    col2.metric("Linhas preparadas", f"{total_limpo:,}".replace(",", "."))
    col3.metric("Duplicatas encontradas", int(duplicadas_brutas))
    col4.metric("Colunas preparadas", len(colunas_convertidas))

    st.subheader("📌 O que foi tratado")
    st.markdown("""
    - `horario` foi convertido para a variável `hora` com formato explícito `%H:%M:%S`, eliminando warnings e deixando o parsing determinístico.
    - `data_inversa` virou data real para permitir leitura mensal e temporal.
    - `km`, `latitude` e `longitude` foram convertidos de texto com vírgula decimal para número.
    - `br` foi padronizada como número inteiro nulo-permitido para preservar o identificador da rodovia.
    - Campos categóricos foram aparados com `strip()` para evitar categorias duplicadas por espaço invisível.
    - Não havia duplicatas na base, então não houve exclusão em massa de linhas.
    """)

    st.subheader("📊 Antes e depois")
    comparacao = pd.DataFrame({
        "Etapa": ["Base bruta", "Base preparada"],
        "Linhas": [total_bruto, total_limpo],
        "Colunas": [df_bruto.shape[1], df.shape[1]],
    })
    st.dataframe(comparacao, use_container_width=True)

    st.subheader("🔍 Amostra da base preparada")
    st.dataframe(
        df[["data_inversa", "hora", "uf", "br", "km", "latitude", "longitude", "causa_acidente", "tipo_acidente"]].head(10),
        use_container_width=True
    )

    st.subheader("🧠 Justificativa técnica")
    st.info(
        "A limpeza foi feita para garantir reprodutibilidade analítica: números deixaram de ser texto, o tempo passou a ser interpretável, e as análises por BR, hora, mês e localização ficaram consistentes."
    )

if menu == "📦 Box Plot e Outliers":

    st.title("📦 Box Plot e Outliers")

    st.markdown("""
    Esta sessão prioriza a leitura de outliers em BRs com maior concentração de acidentes.
    Como apoio, também permite avaliar variáveis numéricas do dataset com a mesma lógica de IQR.
    """)

    if "severidade" not in df.columns:
        df["severidade"] = (
            df["mortos"] * 10 +
            df["feridos_leves"] * 2 +
            df["pessoas"] * 0.5
        )

    aba_br, aba_numerica = st.tabs(["BR com mais acidentes", "Variáveis numéricas"])

    with aba_br:
        st.subheader("📍 Outliers por BR")
        st.caption(
            "Aqui o BR é tratado de forma analítica: a unidade observada é a quantidade de acidentes por rodovia, não o número da BR em si."
        )

        br_df = (
            df.dropna(subset=["br"])
            .groupby("br")
            .size()
            .reset_index(name="total_acidentes")
            .sort_values("total_acidentes", ascending=False)
        )

        if br_df.empty:
            st.warning("Não há dados suficientes de BR para essa análise.")
        else:
            q1 = br_df["total_acidentes"].quantile(0.25)
            q3 = br_df["total_acidentes"].quantile(0.75)
            iqr = q3 - q1
            limite_inferior = q1 - 1.5 * iqr
            limite_superior = q3 + 1.5 * iqr

            outliers_br = br_df[
                (br_df["total_acidentes"] < limite_inferior)
                | (br_df["total_acidentes"] > limite_superior)
            ].copy()

            col1, col2, col3 = st.columns(3)
            col1.metric("🚗 BRs analisadas", len(br_df))
            col2.metric("📦 Mediana de acidentes", round(br_df["total_acidentes"].median(), 2))
            col3.metric("⚠️ BRs outlier", len(outliers_br))

            st.caption(
                f"Limites de outlier para BR: abaixo de {limite_inferior:.2f} ou acima de {limite_superior:.2f} acidentes por rodovia."
            )

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(
                y=br_df["total_acidentes"],
                ax=ax,
                color="#4C78A8",
                width=0.35,
                showfliers=True,
                flierprops={
                    "marker": "o",
                    "markerfacecolor": "#E45756",
                    "markeredgecolor": "#E45756",
                    "markersize": 4,
                    "alpha": 0.8,
                },
            )
            sns.stripplot(
                y=br_df["total_acidentes"],
                ax=ax,
                color="#1f1f1f",
                alpha=0.22,
                size=3,
                jitter=0.18,
            )

            ax.set_xlabel("")
            ax.set_ylabel("Total de acidentes por BR")
            ax.set_title("Distribuição de acidentes por BR com destaque para outliers")
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

            st.subheader("🔎 BRs fora do padrão")
            if outliers_br.empty:
                st.success("Nenhuma BR ficou fora do intervalo esperado pelo critério do IQR.")
            else:
                st.dataframe(
                    outliers_br.rename(columns={"br": "BR", "total_acidentes": "Total de acidentes"})
                )

    with aba_numerica:
        st.subheader("📊 Outliers em variáveis numéricas")

        variaveis_numericas = [
            coluna
            for coluna in ["km", "pessoas", "mortos", "feridos_graves", "feridos_leves", "veiculos", "severidade"]
            if coluna in df.columns
        ]

        variavel = st.selectbox(
            "📌 Variável numérica",
            variaveis_numericas,
            index=0
        )

        dados = df[[variavel]].dropna().copy()
        dados[variavel] = pd.to_numeric(
            dados[variavel].astype(str).str.replace(",", ".", regex=False),
            errors="coerce"
        )
        dados = dados.dropna(subset=[variavel])

        if dados.empty:
            st.warning("Não foi possível converter essa variável para formato numérico.")
            st.stop()

        q1 = dados[variavel].quantile(0.25)
        q3 = dados[variavel].quantile(0.75)
        iqr = q3 - q1
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr

        outliers = dados[(dados[variavel] < limite_inferior) | (dados[variavel] > limite_superior)]

        col1, col2, col3 = st.columns(3)
        col1.metric("📌 Mediana", round(dados[variavel].median(), 2))
        col2.metric("📦 IQR", round(iqr, 2))
        col3.metric("⚠️ Outliers", len(outliers))

        st.caption(
            f"Limites do box plot: abaixo de {limite_inferior:.2f} ou acima de {limite_superior:.2f}"
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(
            y=dados[variavel],
            ax=ax,
            color="#4C78A8",
            width=0.35,
            showfliers=True,
            flierprops={
                "marker": "o",
                "markerfacecolor": "#E45756",
                "markeredgecolor": "#E45756",
                "markersize": 4,
                "alpha": 0.8,
            },
        )
        sns.stripplot(
            y=dados[variavel],
            ax=ax,
            color="#1f1f1f",
            alpha=0.15,
            size=2,
            jitter=0.25
        )
        ax.set_xlabel("")
        ax.set_ylabel(variavel)
        ax.set_title(f"Box plot de {variavel} com destaque para outliers")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        if variavel == "km":
            st.info(
                "Como `km` costuma variar pouco dentro de uma mesma rodovia, ele é útil para achar pontos muito distantes do padrão da base inteira."
            )

if menu == "🔍 Análise Detalhada":

    st.title("🔍 Análise Detalhada de Padrões e Tendências")

    st.markdown("""
    Esta sessão consolida a leitura exploratória em uma visão mais crítica:
    ela mostra concentração de acidentes, comportamento temporal, causas mais recorrentes
    e os pontos onde a base indica maior pressão operacional.
    """)

    df_analise = df.copy()
    df_analise["mes_periodo"] = df_analise["data_inversa"].dt.to_period("M")

    ordem_dias = [
        "segunda-feira", "terça-feira", "quarta-feira",
        "quinta-feira", "sexta-feira", "sábado", "domingo"
    ]

    df_analise = df_analise.dropna(subset=["hora", "mes_periodo"])

    if df_analise.empty:
        st.warning("Não foi possível montar a análise detalhada com os dados disponíveis.")
    else:
        if "severidade" not in df_analise.columns:
            df_analise["severidade"] = (
                df_analise["mortos"] * 10 +
                df_analise["feridos_leves"] * 2 +
                df_analise["pessoas"] * 0.5
            )

        top_brs = (
            df_analise.dropna(subset=["br"])
            .groupby("br")
            .size()
            .sort_values(ascending=False)
        )

        top_uf = df_analise["uf"].value_counts().head(10)
        top_causas = df_analise["causa_acidente"].value_counts().head(10)
        top_tipos = df_analise["tipo_acidente"].value_counts().head(10)

        hora_pico = df_analise["hora"].value_counts().idxmax()
        dia_pico = df_analise["dia_semana"].value_counts().idxmax()
        causa_top = df_analise["causa_acidente"].value_counts().idxmax()
        tipo_top = df_analise["tipo_acidente"].value_counts().idxmax()

        total_acidentes = len(df_analise)
        acidentes_top10_brs = top_brs.head(10).sum() if not top_brs.empty else 0
        concentracao_top10_brs = acidentes_top10_brs / total_acidentes * 100 if total_acidentes else 0
        participacao_top10_causas = top_causas.sum() / total_acidentes * 100 if total_acidentes else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total de acidentes", f"{total_acidentes:,}".replace(",", "."))
        col2.metric("BRs analisadas", df_analise["br"].nunique())
        col3.metric("Hora mais crítica", f"{hora_pico}h")
        col4.metric("Dia mais crítico", dia_pico)

        st.subheader("📌 Leitura executiva")
        st.info(
            f"O acidente é mais concentrado em torno de {causa_top}, com maior pressão em {tipo_top}. "
            f"A distribuição temporal indica pico às {hora_pico}h e maior ocorrência em {dia_pico}."
        )

        if concentracao_top10_brs >= 25:
            st.warning(
                f"As 10 BRs mais incidentes concentram {concentracao_top10_brs:.1f}% dos registros. "
                "Isso sugere forte concentração operacional e priorização clara de fiscalização e engenharia viária."
            )
        else:
            st.success(
                f"A concentração nas 10 BRs mais incidentes é de {concentracao_top10_brs:.1f}%, o que indica distribuição mais espalhada."
            )

        if participacao_top10_causas >= 60:
            st.error(
                f"As 10 principais causas respondem por {participacao_top10_causas:.1f}% dos acidentes. "
                "A base está altamente concentrada em poucos vetores de risco."
            )

        st.subheader("📈 Tendência temporal")
        colA, colB = st.columns(2)

        with colA:
            serie_mensal = df_analise.dropna(subset=["mes_periodo"]).groupby("mes_periodo").size()
            serie_mensal.index = serie_mensal.index.to_timestamp()

            fig1, ax1 = plt.subplots(figsize=(10, 4))
            serie_mensal.plot(ax=ax1, color="#4C78A8", linewidth=2)
            ax1.set_title("Acidentes por mês")
            ax1.set_xlabel("Mês")
            ax1.set_ylabel("Quantidade")
            ax1.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig1)

        with colB:
            hora_dia = df_analise.groupby(["hora", "dia_semana"]).size().unstack().fillna(0)
            hora_dia = hora_dia.reindex(columns=ordem_dias)

            fig2, ax2 = plt.subplots(figsize=(10, 4))
            sns.heatmap(hora_dia.T, cmap="Blues", ax=ax2)
            ax2.set_title("Distribuição de acidentes por hora e dia")
            ax2.set_xlabel("Hora do dia")
            ax2.set_ylabel("Dia da semana")
            plt.tight_layout()
            st.pyplot(fig2)

        st.subheader("🧭 Concentração territorial e causal")
        colC, colD = st.columns(2)

        with colC:
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            top_uf.plot(kind="bar", ax=ax3, color="#E45756")
            ax3.set_title("Top 10 UFs com maior número de acidentes")
            ax3.set_xlabel("UF")
            ax3.set_ylabel("Quantidade")
            ax3.tick_params(axis="x", rotation=0)
            ax3.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig3)

        with colD:
            fig4, ax4 = plt.subplots(figsize=(10, 4))
            top_causas.head(10).sort_values().plot(kind="barh", ax=ax4, color="#72B7B2")
            ax4.set_title("Top 10 causas de acidente")
            ax4.set_xlabel("Quantidade")
            ax4.set_ylabel("")
            ax4.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig4)

        st.subheader("🚧 Concentração por BR")
        if top_brs.empty:
            st.info("Não há dados de BR suficientes para detalhar a concentração.")
        else:
            br_top_df = top_brs.head(15).reset_index()
            br_top_df.columns = ["br", "total_acidentes"]

            fig5, ax5 = plt.subplots(figsize=(10, 5))
            br_top_df.sort_values("total_acidentes").plot(
                kind="barh",
                x="br",
                y="total_acidentes",
                ax=ax5,
                color="#4C78A8",
                legend=False,
            )
            ax5.set_title("Top 15 BRs com mais acidentes")
            ax5.set_xlabel("Quantidade")
            ax5.set_ylabel("BR")
            ax5.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig5)

            st.write(
                f"As 15 BRs mais críticas concentram {br_top_df['total_acidentes'].sum() / total_acidentes * 100:.1f}% dos registros da base."
            )

        st.subheader("🧠 Interpretação crítica")
        st.markdown("""
        - A maior parte dos acidentes se organiza em poucos pontos de pressão: horário de pico, causas recorrentes e rodovias mais incidentes.
        - Quando o volume de acidentes se concentra em um grupo pequeno de BRs, a resposta mais eficiente costuma ser localizada, não apenas geral.
        - Se a distribuição por UF for muito desigual, isso pode indicar diferença de malha, volume de tráfego, fiscalização ou qualidade viária.
        - O fato de a base estar concentrada em causas repetidas é um sinal de oportunidade de prevenção com medidas operacionais simples, e não apenas ações estruturais de longo prazo.
        """)

        with st.expander("📋 Resumo automático dos achados"):
            st.write(f"Hora de maior incidência: {hora_pico}h")
            st.write(f"Dia de maior incidência: {dia_pico}")
            st.write(f"Causa mais frequente: {causa_top}")
            st.write(f"Tipo de acidente mais frequente: {tipo_top}")
            st.write(f"Concentração das 10 BRs mais incidentes: {concentracao_top10_brs:.1f}%")
            st.write(f"Participação das 10 principais causas: {participacao_top10_causas:.1f}%")

if menu == "🧩 Perfil Operacional da Via":

    st.title("🧩 Perfil Operacional da Via")
    st.caption("Sessão complementar para explorar contexto viário e operacional além das análises clássicas.")

    st.markdown("""
    Esta análise foca em variáveis de ambiente e operação da via que também estão no CSV:
    - uso do solo
    - tipo e traçado da pista
    - sentido da via
    - condição meteorológica
    - regional e delegacia
    """)

    if "severidade" not in df.columns:
        df["severidade"] = criar_severidade(df)

    visao_uf = st.selectbox("Filtrar UF (opcional)", ["Todos"] + sorted(df["uf"].dropna().unique()))
    base = df.copy()
    if visao_uf != "Todos":
        base = base[base["uf"] == visao_uf]

    if base.empty:
        st.warning("Sem dados para o filtro selecionado.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registros", f"{len(base):,}".replace(",", "."))
    col2.metric("Tipos de pista", base["tipo_pista"].nunique())
    col3.metric("Traçados de via", base["tracado_via"].nunique())
    col4.metric("Condições meteo", base["condicao_metereologica"].nunique())

    st.subheader("🌦️ Condição meteorológica e severidade")
    meteo = (
        base.groupby("condicao_metereologica")
        .agg(
            acidentes=("id", "count"),
            severidade_media=("severidade", "mean"),
        )
        .sort_values("acidentes", ascending=False)
        .head(12)
        .reset_index()
    )

    colA, colB = st.columns(2)
    with colA:
        fig1, ax1 = plt.subplots(figsize=(9, 4))
        meteo.sort_values("acidentes").plot(
            kind="barh",
            x="condicao_metereologica",
            y="acidentes",
            color="#4C78A8",
            legend=False,
            ax=ax1,
        )
        ax1.set_title("Top condições meteorológicas por volume")
        ax1.set_xlabel("Quantidade de acidentes")
        ax1.set_ylabel("")
        ax1.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1)

    with colB:
        fig2, ax2 = plt.subplots(figsize=(9, 4))
        meteo.sort_values("severidade_media").plot(
            kind="barh",
            x="condicao_metereologica",
            y="severidade_media",
            color="#E45756",
            legend=False,
            ax=ax2,
        )
        ax2.set_title("Severidade média por condição meteorológica")
        ax2.set_xlabel("Severidade média")
        ax2.set_ylabel("")
        ax2.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)

    st.subheader("🛣️ Estrutura da via")
    estrutura = base.groupby(["tipo_pista", "tracado_via"]).size().unstack().fillna(0)
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.heatmap(estrutura, cmap="YlGnBu", ax=ax3)
    ax3.set_title("Tipo de pista x Traçado da via")
    ax3.set_xlabel("Traçado da via")
    ax3.set_ylabel("Tipo de pista")
    plt.tight_layout()
    st.pyplot(fig3)

    st.subheader("↔️ Sentido da via e uso do solo")
    colC, colD = st.columns(2)

    with colC:
        sentido = base["sentido_via"].value_counts().head(10)
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        sentido.sort_values().plot(kind="barh", color="#72B7B2", ax=ax4)
        ax4.set_title("Distribuição por sentido da via")
        ax4.set_xlabel("Quantidade")
        ax4.set_ylabel("")
        ax4.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4)

    with colD:
        uso_solo = base["uso_solo"].value_counts().head(10)
        fig5, ax5 = plt.subplots(figsize=(8, 4))
        uso_solo.sort_values().plot(kind="barh", color="#F58518", ax=ax5)
        ax5.set_title("Distribuição por uso do solo")
        ax5.set_xlabel("Quantidade")
        ax5.set_ylabel("")
        ax5.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig5)

    st.subheader("🏢 Regional e delegacia")
    dimensao_gestao = st.selectbox("Agrupar gestão por", ["regional", "delegacia", "uop"])
    tabela_gestao = (
        base.groupby(dimensao_gestao)
        .agg(
            acidentes=("id", "count"),
            severidade_media=("severidade", "mean")
        )
        .sort_values("acidentes", ascending=False)
        .head(20)
        .reset_index()
    )
    st.dataframe(
        tabela_gestao.rename(
            columns={
                dimensao_gestao: dimensao_gestao.capitalize(),
                "acidentes": "Total de acidentes",
                "severidade_media": "Severidade média"
            }
        ),
        width="stretch"
    )

    st.subheader("🎯 Leitura prática")
    meteo_top = meteo.iloc[0]["condicao_metereologica"] if not meteo.empty else "N/A"
    pista_top = base["tipo_pista"].value_counts().idxmax() if base["tipo_pista"].notna().any() else "N/A"
    solo_top = base["uso_solo"].value_counts().idxmax() if base["uso_solo"].notna().any() else "N/A"

    st.info(
        f"Maior concentração operacional em: condição meteorológica '{meteo_top}', "
        f"tipo de pista '{pista_top}' e uso do solo '{solo_top}'."
    )

    st.markdown("""
    **Sugestões de ação com base no perfil operacional:**
    - priorizar fiscalização e sinalização nos perfis de via com maior concentração;
    - ajustar operação em janelas climáticas de maior ocorrência;
    - cruzar regional/delegacia com severidade média para alocar recursos de forma mais eficiente.
    """)