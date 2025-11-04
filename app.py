# app.py

import streamlit as st
import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv

# --- Configuração de Credenciais ---

# 1. Carrega as variáveis de ambiente do arquivo .env
# Isso é necessário para o desenvolvimento local.
# No Streamlit Cloud, ele usará o Secrets Management nativo.
load_dotenv() 

# Função para buscar os dados de forma segura e cacheada no Streamlit
# O decorador 'st.cache_data' garante que a conexão e a busca ocorram
# apenas uma vez ou quando a função mudar.
@st.cache_data(show_spinner="Conectando ao banco de dados e carregando dados...")
def get_data_from_db():
    
    # 2. Obtém as credenciais das variáveis de ambiente
    DB_HOST = os.getenv("PG_HOST")
    DB_PORT = os.getenv("PG_PORT")
    DB_NAME = os.getenv("PG_DATABASE")
    DB_USER = os.getenv("PG_USER")
    DB_PASSWORD = os.getenv("PG_PASSWORD")

    # Sua consulta SQL para buscar os dados de predição
    SQL_QUERY = """
        select 
            fm.ano as "Ano",
            fm.uf as "Unidade Federativa",
            fm.municipio as "Município",
            fm.total as "Total veículos",
            pm.vl_pib as "Valor PIB"
        from frota_municipios fm 
        left join pib_municipios pm 
            on fm.codigo_ibge = pm.codigo_municipio_dv and fm.ano = cast(pm.ano_pib as integer)
        where fm.ano <= 2020 and cast(pm.ano_pib as integer) <= 2020
    """

    conn = None # Inicializa a conexão
    try:
        # 3. Estabelece a conexão com o psycopg2
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        st.success("Conexão com o PostgreSQL estabelecida com sucesso!")
        
        # 4. Lê os dados para um DataFrame do Pandas
        df = pd.read_sql(SQL_QUERY, conn)
        
        return df

    except Exception as e:
        st.error(f"Erro ao conectar ou buscar dados: {e}")
        st.stop() # Interrompe a execução do Streamlit em caso de falha
    
    finally:
        # 5. Fecha a conexão
        if conn:
            conn.close()

# --- Execução Principal do App ---

st.title("🚗 Projeto de Predição de Volume de Veículos")

# Carrega os dados
df_dados = get_data_from_db()

# Exibe uma amostra (para verificar se o carregamento funcionou)
st.subheader("Amostra dos Dados Carregados")
st.dataframe(df_dados.head())

# st.write(f"Total de linhas carregadas: {len(df_dados)}")

import plotly.express as px
import plotly.graph_objects as go # Usaremos para um gráfico mais detalhado

# --- Análise Exploratória de Dados (EDA) ---
st.subheader("📊 Análise Exploratória de Dados (EDA)")

# 1. Distribuição da Variável Alvo: Volume de Veículos
st.markdown("#### 1. Distribuição do Volume de Veículos")
st.info("A distribuição da variável alvo é crucial para a modelagem bayesiana, pois ela informa a escolha da distribuição de probabilidade (likelihood) na sua inferência.")

fig_hist = px.histogram(
    df_dados, 
    x='volume_veiculos', 
    nbins=50, 
    title='Histograma da Variável Alvo: Volume de Veículos'
)
fig_hist.update_layout(bargap=0.1) # Adiciona um pequeno espaço entre as barras
st.plotly_chart(fig_hist, use_container_width=True)


# 2. Relação da Variável Alvo com a Variável Numérica (PIB)
st.markdown("#### 2. Relação entre Volume de Veículos e PIB")

fig_scatter = px.scatter(
    df_dados, 
    x='pib_valor', 
    y='volume_veiculos', 
    color='unidade_federativa', # Colore pelo estado para adicionar contexto
    opacity=0.6,
    log_x=True, # Aplica escala logarítmica ao PIB, pois a distribuição costuma ser assimétrica
    title='Volume de Veículos vs. Valor do PIB (Por UF)'
)
st.plotly_chart(fig_scatter, use_container_width=True)


# 3. Relação da Variável Alvo com Variáveis Categóricas (Ano e UF)
st.markdown("#### 3. Volume de Veículos por Unidade Federativa (UF) e Ano")

# Gráfico de Boxplot para Volume por UF
fig_box_uf = px.box(
    df_dados,
    x='unidade_federativa',
    y='volume_veiculos',
    title='Distribuição do Volume de Veículos por UF',
    notched=True # Adiciona recortes para indicar diferenças estatísticas (aproximadas)
)
st.plotly_chart(fig_box_uf, use_container_width=True)

# Gráfico de Linha para Tendência Temporal (Volume Médio por Ano)
df_trend = df_dados.groupby('ano')['volume_veiculos'].mean().reset_index()
fig_line_year = px.line(
    df_trend,
    x='ano',
    y='volume_veiculos',
    title='Tendência do Volume Médio de Veículos ao Longo dos Anos',
    markers=True
)
st.plotly_chart(fig_line_year, use_container_width=True)

# --- Fim da Seção EDA ---

# st.markdown("## 🔮 Inferência Bayesiana e Predição (Próxima Etapa)")