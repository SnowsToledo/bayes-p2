# app.py

import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
import os
import arviz as az
from dotenv import load_dotenv
from scipy.stats import gaussian_kde # Para estimar a densidade

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
    x='Total veículos', 
    nbins=50, 
    title='Histograma da Variável Alvo: Volume de Veículos'
)
fig_hist.update_layout(bargap=0.1) # Adiciona um pequeno espaço entre as barras
st.plotly_chart(fig_hist, use_container_width=True)


# 2. Relação da Variável Alvo com a Variável Numérica (PIB)
st.markdown("#### 2. Relação entre Volume de Veículos e PIB")

fig_scatter = px.scatter(
    df_dados, 
    x='Valor PIB', 
    y='Total veículos', 
    color='Unidade Federativa', # Colore pelo estado para adicionar contexto
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
    x='Unidade Federativa',
    y='Total veículos',
    title='Distribuição do Volume de Veículos por UF',
    notched=True # Adiciona recortes para indicar diferenças estatísticas (aproximadas)
)
st.plotly_chart(fig_box_uf, use_container_width=True)

# Gráfico de Linha para Tendência Temporal (Volume Médio por Ano)
df_trend = df_dados.groupby('Ano')['Total veículos'].mean().reset_index()
fig_line_year = px.line(
    df_trend,
    x='Ano',
    y='Total veículos',
    title='Tendência do Volume Médio de Veículos ao Longo dos Anos',
    markers=True
)
st.plotly_chart(fig_line_year, use_container_width=True)

# --- Fim da Seção EDA ---
df_transformado = df_dados.copy()

df_transformado['log_Total_veículos'] = np.log(df_transformado['Total veículos'])
df_transformado['log_Valor_PIB'] = np.log(df_transformado['Valor PIB'])

# st.markdown("## 🔮 Inferência Bayesiana e Predição (Próxima Etapa)")
import pymc as pm

# 1. Aplicar a transformação log nos dados antes de criar o modelo
y_obs = df_transformado['log_Total_veículos']
X_obs = df_transformado['log_Valor_PIB']

@st.cache_resource
def rodar_modelo_bayesiano(y_obs, X_obs):
    """Função para construir e rodar o modelo PyMC."""
    with pm.Model() as modelo_bayesiano:
            # Priores (exemplo)
        alfa = pm.Normal('alfa', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=2)
        sigma = pm.HalfCauchy('sigma', beta=1)

        # Modelo Linear
        mu = alfa + beta * X_obs
        # Amostragem
        traço = pm.sample(2000, tune=1000, return_inferencedata=True)
        return traço

# Carregamento e transformação dos dados (pode ser feito com st.cache_data)
# ...

# 3. Rodar o modelo e obter o traço (o Streamlit só roda a amostragem uma vez)
traco_cacheado = rodar_modelo_bayesiano(y_obs, X_obs)
# Exemplo de visualização no Streamlit
st.header("Análise Bayesiana dos Resultados")
def plot_trace_direct_plotly(traco, param_name):
    """
    Cria um Gráfico de Traço (Trace Plot) usando Plotly.
    param_name: nome do parâmetro (string, ex: 'beta')
    """
    posterior_data = traco.posterior[param_name]
    n_chains = posterior_data.sizes['chain']
    n_draws = posterior_data.sizes['draw']
    
    fig = go.Figure()
    
    for chain in range(n_chains):
        # Seleciona as amostras para a cadeia atual
        samples = posterior_data.sel(chain=chain).values.flatten()
        
        fig.add_trace(go.Scatter(
            x=np.arange(n_draws), 
            y=samples,
            mode='lines',
            name=f'Cadeia {chain + 1}',
            line={'width': 1}
        ))
        
    fig.update_layout(
        title=f'Traço MCMC para o Parâmetro: {param_name}',
        xaxis_title='Passo da Amostragem',
        yaxis_title='Valor do Parâmetro',
        height=400,
        hovermode="x unified"
    )
    return fig

# Exemplo de Uso no Streamlit:
st.header("📈 Convergência do Parâmetro Beta")
fig_trace_conv_beta = plot_trace_direct_plotly(traco_cacheado, 'beta')
st.plotly_chart(fig_trace_conv_beta, use_container_width=True)

# Exemplo de Uso no Streamlit:
st.header("📈 Convergência do Parâmetro Alfa")
fig_trace_conv_alfa = plot_trace_direct_plotly(traco_cacheado, 'alfa')
st.plotly_chart(fig_trace_conv_alfa, use_container_width=True)

def plot_posterior_direct_plotly(traco, param_name):
    """
    Cria um Gráfico de Densidade Posterior (KDE) usando Plotly.
    param_name: nome do parâmetro (string, ex: 'beta')
    """
    # Combina todas as amostras (cadeias e passos) em um único array
    all_samples = traco.posterior[param_name].values.flatten()
    
    # Usa Plotly Express para criar um Histograma e estimativa de Densidade (KDE)
    fig = px.histogram(
        all_samples, 
        nbins=50, 
        marginal="box", # Adiciona um box plot marginal para resumo
        histnorm='probability density', # Normaliza para densidade
        title=f'Distribuição a Posteriori do Parâmetro: {param_name}'
    )
    
    # Opcional: Adicionar a linha KDE (se não usar o marginal do px)
    # kde = gaussian_kde(all_samples)
    # x_vals = np.linspace(all_samples.min(), all_samples.max(), 500)
    # fig.add_trace(go.Scatter(x=x_vals, y=kde(x_vals), mode='lines', name='KDE', line={'color': 'red'}))
    
    # Adicionar a linha da Média/Mediana (estimativa pontual)
    median_val = np.median(all_samples)
    fig.add_vline(
        x=median_val, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Mediana: {median_val:.3f}", 
        annotation_position="top right"
    )

    fig.update_layout(
        showlegend=False,
        yaxis_title="Densidade de Probabilidade",
        xaxis_title=f"Valor de {param_name}",
        height=400
    )
    return fig

# Exemplo de Uso no Streamlit:
st.header("🔬 Densidade Posterior do Parâmetro Beta")
fig_posterior_dens_beta = plot_posterior_direct_plotly(traco_cacheado, 'beta')
st.plotly_chart(fig_posterior_dens_beta, use_container_width=True)

st.header("🔬 Densidade Posterior do Parâmetro Alfa")
fig_posterior_dens_alfa = plot_posterior_direct_plotly(traco_cacheado, 'alfa')
st.plotly_chart(fig_posterior_dens_alfa, use_container_width=True)