# app.py

import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
import os
import arviz as az
from dotenv import load_dotenv
from scipy.stats import gaussian_kde # Para estimar a densidade
import pymc as pm
import plotly.express as px
import plotly.graph_objects as go # Usaremos para um gráfico mais detalhado

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

st.write(f"Total de linhas carregadas: {len(df_dados)}")

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
df_transformado['mun_numerico'] = df_transformado['Município'].astype('category').cat.codes


# st.markdown("## 🔮 Inferência Bayesiana e Predição (Próxima Etapa)")


# 1. Aplicar a transformação log nos dados antes de criar o modelo
y_obs = df_transformado['log_Total_veículos']
X_obs_1 = df_transformado['log_Valor_PIB']
X_obs_2 = df_transformado['mun_numerico'].values
N_Municipios = df_transformado['mun_numerico'].nunique()
mun_names = df_transformado['Município'].unique().tolist()

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
        traco = pm.sample(2000, tune=1000, return_inferencedata=True)
        return traco

@st.cache_resource
def rodar_modelo_bayesiano_multivariado(y_obs, X_obs_1, X_obs_2, N_Municipios):
    # 2. Definição do Modelo Hierárquico no PyMC
    with pm.Model() as hierarchical_model:
        
        # ---- Hiper-priors (Nível 2) ----
        
        # Média e desvio padrão globais para os interceptos
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
        tau_alpha = pm.HalfCauchy("tau_alpha", beta=1)
        
        # Distribuição dos Interceptos de cada Município
        # pm.Normal.dist é uma 'distribuição de distribuição'
        alpha = pm.Normal("alpha", mu=mu_alpha, sigma=tau_alpha, shape=N_Municipios)
        
        # Coeficiente do PIB (Global, não-hierárquico neste modelo)
        beta_PIB = pm.Normal("beta_PIB", mu=0, sigma=10)
        
        # Desvio padrão residual (Global)
        sigma = pm.HalfCauchy("sigma", beta=1)
        
        # ---- Média Linear (Nível 1) ----
        
        # O intercepto é específico para cada município (alpha[mun_idx_data])
        # O coeficiente do PIB é o mesmo para todos (beta_PIB)
        mu = alpha[X_obs_2] + beta_PIB * X_obs_1
        
        # ---- Likelihood (Verossimilhança) ----
        
        # Os dados observados
        Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=y_obs)
        
        # 3. Inferência (Amostragem MCMC)
        st.write("Rodando Inferência Bayesiana (MCMC)...")
        traco = pm.sample(2000, tune=1000, cores=2, return_idata=True)
    return traco


# Carregamento e transformação dos dados (pode ser feito com st.cache_data)
# ...

# 3. Rodar o modelo e obter o traço (o Streamlit só roda a amostragem uma vez)
traco_cacheado = rodar_modelo_bayesiano(y_obs, X_obs_1)

traco_cacheado_multivariado = rodar_modelo_bayesiano_multivariado(y_obs, X_obs_1, X_obs_2, N_Municipios)
st.header("Análise Bayesiana")

st.write("""**Modelo Bayesiano escolhido**: Regressão Linear Hierárquica Bayesiana com Interceptos Variáveis (Efeitos Aleatórios).\n
         **Justificativa**: \n
         - Estrutura Agrupada dos Dados: Os dados de PIB e Volume de Veículos estão agrupados por Município. Ignorar essa estrutura (usando uma Regressão Linear Simples) violaria a suposição de independência das observações, pois municípios que pertencem ao mesmo estado ou região podem ter características de tráfego mais similares entre si.
         - Controle de Heterogeneidade Não Observada (Efeito do Município): O volume base de veículos pode ser influenciado por fatores que não estão no modelo (ex: ser capital, estar em uma rota comercial principal, políticas de transporte, topografia). O intercepto variável, $\alpha_j$, absorve essas diferenças de "nível" para cada município $j$, sem a necessidade de incluir inúmeras variáveis dummy na regressão.
         - "Pooling" de Informação: O mecanismo hierárquico permite que municípios com poucos dados ("pequenos") se beneficiem da informação dos municípios com muitos dados ("grandes"). Isso leva a estimativas ($\alpha_j$) mais estáveis e menos extremas, um fenômeno conhecido como shrinkage (encolhimento).
         - Quantificação de Incerteza: A abordagem Bayesiana fornece uma distribuição completa (Posteriori) para os parâmetros, permitindo a quantificação da incerteza nas estimativas de forma mais intuitiva do que as estatísticas frequentistas (Intervalos de Credibilidade vs. Intervalos de Confiança).
         **Modelo Estatístico**:\n
         O modelo estatístico hierárquico é definido em dois níveis:\n
         Nível 1: Modelo de Dados (Likelihood)\n
         Define a relação entre o Volume de Veículos ($Y$) e o PIB ($X$) para o município $j$ na observação $i$:\n
         $$Y_{ij} \sim \mathcal{N}(\mu_{ij}, \sigma^2)$$\n
         $$\mu_{ij} = \alpha_j + \beta \cdot \text{PIB}_{ij}$$\n
         - $Y_{ij}$: Volume de Veículos observado.\n
         - $\text{PIB}_{ij}$: Produto Interno Bruto (PIB) municipal.\n
         - $\sigma$: Desvio padrão (incerteza residual), assumido comum para todos os municípios.\n
         - $\alpha_j$: Intercepto (Efeito Aleatório) específico do Município $j$.\n
         - $\beta$: Coeficiente de Regressão, assumido fixo (igual) para todos os municípios.\n
         Nível 2: Modelo Hierárquico (Priors/Hiper-Priors)\n
         Define como os parâmetros do Nível 1 estão distribuídos:
         - Interceptos (Efeitos Aleatórios): Os interceptos de cada município são modelados como vindos de uma distribuição Normal comum:\n
         $$\alpha_j \sim \mathcal{N}(\mu_{\alpha}, \tau^2)$$
""")

# 4. Análise dos Resultados (Exemplo)
# Sumário estatístico dos parâmetros
summary = pm.summary(traco_cacheado_multivariado, var_names=['mu_alpha', 'tau_alpha', 'beta_PIB'])
st.subheader("\nSumário de Parâmetros Globais")
st.write(summary)

# Exemplo de visualização no Streamlit

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

st.write("""
    **Distribuições A Priori (Priors)**\n
    Os priors selecionados são considerados Priors Fracamente Informativos para permitir que os dados (Likelihood) dominem a inferência, ao mesmo tempo que evitam distribuições problemáticas (como a Uniforme sobre um domínio infinito).
         1. $\beta$ e $\mu_{\alpha} \sim \mathcal{N}(0, 10)$:\n
         - Justificativa: A Normal com média zero e desvio padrão 10 é uma escolha padrão para coeficientes de regressão. Ela é centrada em zero (nenhum efeito a priori) e possui desvio padrão grande o suficiente para cobrir um vasto intervalo de valores plausíveis para o coeficiente e a média dos interceptos.\n
         2. $\tau$ e $\sigma \sim \text{HalfCauchy}(1)$:
         - Justificativa: A Half-Cauchy é ideal para parâmetros de escala (desvios padrões, que devem ser positivos). Ela é centrada em zero e é "long-tailed" (possui caudas pesadas), permitindo que os desvios padrões globais $\tau$ e $\sigma$ assumam valores grandes se os dados assim indicarem, mas concentra a maior parte da massa de probabilidade em valores menores.
""")



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

st.write("""
    **Distribuições A Posteriori (Posteriors)**\n
    As distribuições a posteriori são obtidas após a execução do algoritmo MCMC (Método de Monte Carlo por Cadeias de Markov) e representam o nosso conhecimento atualizado sobre os parâmetros após a observação dos dados.\n
    - $\text{Posteriori}(\mu_{\alpha}, \tau, \beta, \sigma, \alpha_j \mid Y, \text{PIB}) \propto \text{Likelihood}(\ldots) \times \text{Priors}(\ldots)$
""")
# Exemplo de Uso no Streamlit:
st.header("🔬 Densidade Posterior do Parâmetro Beta")
fig_posterior_dens_beta = plot_posterior_direct_plotly(traco_cacheado, 'beta')
st.plotly_chart(fig_posterior_dens_beta, use_container_width=True)

st.header("🔬 Densidade Posterior do Parâmetro Alfa")
fig_posterior_dens_alfa = plot_posterior_direct_plotly(traco_cacheado, 'alfa')
st.plotly_chart(fig_posterior_dens_alfa, use_container_width=True)

def hdi_manual(posterior_samples, hdi_prob=0.95):
    """Calcula o Highest Density Interval (HDI) para amostras."""
    samples = np.sort(posterior_samples)
    n = len(samples)
    interval_size = int(np.floor(n * hdi_prob))
    if interval_size == 0:
        return samples[0], samples[-1]

    intervals = samples[interval_size:] - samples[:n - interval_size]
    min_idx = np.argmin(intervals)
    return samples[min_idx], samples[min_idx + interval_size]

# ====================================================================
# FUNÇÃO PLOTLY SEM ARVIZ
# ====================================================================

def plot_intercepts_plotly_manual(trace, mun_names):
    """Calcula a média e HDI dos interceptos e plota com Plotly."""
    
    # 1. Extrair amostras dos interceptos (shape: chains, draws, N_MUN)
    alpha_samples = trace.posterior['alpha'].values.reshape(-1, len(mun_names))
    
    results = []
    
    # 2. Iterar sobre cada município para calcular Média e HDI
    for i, mun_name in enumerate(mun_names):
        samples_i = alpha_samples[:, i]
        
        # Cálculo Manual da Média e HDI
        mean = np.mean(samples_i)
        hdi_lower, hdi_upper = hdi_manual(samples_i, hdi_prob=0.95)
        
        results.append({
            'Município': mun_name,
            'mean': mean,
            'hdi_2.5%': hdi_lower,
            'hdi_97.5%': hdi_upper
        })

    hdi_df = pd.DataFrame(results)
    
    # 3. Calcular a Média Global dos Interceptos (mu_alpha)
    mu_alpha_mean = trace.posterior['mu_alpha'].mean().item()
    
    # 4. Criar o gráfico Plotly
    fig = go.Figure()
    
    # Adicionar as barras de erro (Intervalo de Credibilidade de 95%)
    fig.add_trace(go.Scatter(
        x=hdi_df['mean'],
        y=hdi_df['Município'],
        mode='markers',
        error_x=dict(
            type='data',
            symmetric=False,
            # Calcula a diferença do valor da Média para os limites do HDI
            array=hdi_df['hdi_97.5%'] - hdi_df['mean'],
            arrayminus=hdi_df['mean'] - hdi_df['hdi_2.5%'],
            thickness=1.5,
            width=5
        ),
        marker=dict(size=8, color='darkblue'),
        name='Média Posterior com HDI 95%'
    ))
    
    # Adicionar a linha da Média Global (μ_α)
    fig.add_shape(
        type='line',
        x0=mu_alpha_mean,
        x1=mu_alpha_mean,
        y0=-0.5,
        y1=len(mun_names) - 0.5,
        line=dict(color='red', width=2, dash='dash'),
        name=f'Média Global ({mu_alpha_mean:.2f})'
    )
    
    # 5. Configurar Layout
    fig.update_layout(
        title='Distribuição Posterior dos Interceptos por Município (αⱼ)',
        xaxis_title='Intercepto (Volume Base de Veículos)',
        yaxis_title='',
        height=700,
        showlegend=True
    )
    
    return fig

# --- STREAMLIT APP ---

st.title("🚗 Variação Intermunicipal no Volume de Veículos (Modelo Hierárquico)")

st.subheader("1. Coeficiente Global do PIB")
# Cálculo manual da média do beta_PIB
beta_pib_mean = traco_cacheado_multivariado.posterior['beta_PIB'].mean().item()
st.metric(label="Impacto Médio do PIB (β)", value=f"{beta_pib_mean:.4f}")
st.markdown(f"> O Volume de Veículos aumenta em **{beta_pib_mean:.4f}** unidades, em média, para cada unidade de aumento no PIB.")

st.subheader("2. Efeitos Aleatórios por Município (αⱼ) - Visualização Plotly")

# Chama a função para gerar o gráfico (usando cálculo manual)
fig_intercepts = plot_intercepts_plotly_manual(traco_cacheado_multivariado, mun_names)

# Exibe o gráfico no Streamlit
st.plotly_chart(fig_intercepts, use_container_width=True)

st.markdown(r"""
### Análise do Gráfico:
* **Ponto Azul:** Média Posterior do **Intercepto ($\alpha_j$)** para cada município.
* **Barra Horizontal:** **Intervalo de Credibilidade de 95% (HDI)**, representando a incerteza.
* **Linha Tracejada Vermelha:** A **Média Global dos Interceptos ($\mu_{\alpha}$)**, que serve como referência para o grupo.
""")


def plot_predictions(trace, df, selected_mun_name):
    """
    Gera as predições e plota o resultado com Plotly, comparando a
    regressão individual com a média global.
    """
    
    # 1. Definir o range de PIB para as predições
    pib_range = np.linspace(df['log_Valor_PIB'].min(), df['log_Valor_PIB'].max(), 100)
    
    # 2. Extrair amostras dos parâmetros
    alpha_samples = trace.posterior['alpha'].values.reshape(-1, N_Municipios)
    mu_alpha_samples = trace.posterior['mu_alpha'].values.flatten()
    beta_samples = trace.posterior['beta_PIB'].values.flatten()
    
    # 3. Predição Global (Usando a Média Global dos Interceptos: mu_alpha)
    # Calcule a linha de regressão para cada amostra da MCMC
    global_predictions = np.outer(pib_range, beta_samples) + mu_alpha_samples
    
    # Calcule a Média e o HDI da Predição Global
    global_mean = np.mean(global_predictions, axis=1)
    global_hdi_lower = np.array([hdi_manual(global_predictions[i, :])[0] for i in range(len(pib_range))])
    global_hdi_upper = np.array([hdi_manual(global_predictions[i, :])[1] for i in range(len(pib_range))])

    # 4. Predição Individual do Município Selecionado
    selected_mun_idx = df_transformado[df_transformado['Município'] == selected_mun_name]['mun_numerico'].iloc[0]
    
    # Extrai as amostras de alpha específicas para o município selecionado
    mun_alpha_samples = alpha_samples[:, selected_mun_idx]
    
    # Calcule a linha de regressão para cada amostra da MCMC (Individual)
    mun_predictions = np.outer(pib_range, beta_samples) + mun_alpha_samples
    
    # Calcule a Média e o HDI da Predição Individual
    mun_mean = np.mean(mun_predictions, axis=1)
    mun_hdi_lower = np.array([hdi_manual(mun_predictions[i, :])[0] for i in range(len(pib_range))])
    mun_hdi_upper = np.array([hdi_manual(mun_predictions[i, :])[1] for i in range(len(pib_range))])

    # 5. Criação do Plotly Figure
    fig = go.Figure()

    # --- Plot da Predição Global (Referência) ---
    
    # Intervalo de Credibilidade (Sombra/Faixa) Global
    fig.add_trace(go.Scatter(
        x=np.concatenate([pib_range, pib_range[::-1]]), # Liga x0 com x1
        y=np.concatenate([global_hdi_upper, global_hdi_lower[::-1]]), # Liga y_upper com y_lower
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.1)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        name='HDI 95% Global'
    ))
    
    # Linha Média Global
    fig.add_trace(go.Scatter(
        x=pib_range, 
        y=global_mean, 
        mode='lines', 
        line=dict(color='red', dash='dash'),
        name='Média Global'
    ))

    # --- Plot da Predição Individual (Município Selecionado) ---
    
    # Intervalo de Credibilidade (Sombra/Faixa) Individual
    fig.add_trace(go.Scatter(
        x=np.concatenate([pib_range, pib_range[::-1]]),
        y=np.concatenate([mun_hdi_upper, mun_hdi_lower[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.15)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        name=f'HDI 95% {selected_mun_name}'
    ))
    
    # Linha Média Individual
    fig.add_trace(go.Scatter(
        x=pib_range, 
        y=mun_mean, 
        mode='lines', 
        line=dict(color='blue', width=3),
        name=f'Predição para {selected_mun_name}'
    ))
    
    # --- Configurações do Layout ---
    
    fig.update_layout(
        title=f'Predição do Volume de Veículos vs. PIB: {selected_mun_name}',
        xaxis_title='PIB Municipal',
        yaxis_title='Volume de Veículos (Predito)',
        hovermode='x unified'
    )
    
    return fig

# ====================================================================
# APP STREAMLIT
# ====================================================================

st.title("🔮 Predições do Modelo Hierárquico Bayesiano")
st.markdown("Compare a previsão para o município selecionado com o efeito médio global.")

# Seletor de Município
unique_mun_names = sorted(df_transformado['Município'].unique())
selected_mun = st.selectbox(
    "Selecione um Município para a Predição:",
    unique_mun_names
)

if selected_mun:
    # Chama a função de plotagem e predição
    fig_predictions = plot_predictions(traco_cacheado_multivariado, df_transformado, selected_mun)
    st.plotly_chart(fig_predictions, use_container_width=True)

    st.subheader("Interpretação da Predição")
    st.markdown(r"""
    * A **Linha Azul** e a **Faixa Azul** representam a **predição específica** para o município selecionado, incorporando seu **Intercepto ($\alpha_j$)** único.
    * A **Linha Tracejada Vermelha** e a **Faixa Vermelha** representam a **Média Global ($\mu_{\alpha}$)** do grupo, ignorando a identidade individual do município.
    * **HDI 95% (Faixa Sombreada):** Indica que há 95% de chance de o valor real do Volume de Veículos cair dentro dessa faixa, para um dado valor de PIB.
    """)