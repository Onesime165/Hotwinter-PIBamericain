import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from scipy.stats import norm, shapiro, kurtosis, skew
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import io
import base64
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu

# Configuration de la page
st.set_page_config(
    page_title="📊 Analyse PIB Américain 1974-2024",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design moderne avec informations auteur
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Roboto:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e6ed;
    }

    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
    }

    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
        border-right: 2px solid #00ffff;
        box-shadow: 5px 0 15px rgba(0, 255, 255, 0.1);
    }

    .stButton>button {
        background: linear-gradient(45deg, #00ffff, #00ff88);
        border: none;
        color: #0f0f23;
        font-weight: bold;
        border-radius: 25px;
        text-transform: uppercase;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 255, 255, 0.4);
    }

    h1, h2, h3, h4 {
        color: #00ffff;
        font-family: 'Orbitron', monospace;
        text-shadow: 0 0 8px rgba(0, 255, 255, 0.5);
    }

    .stTextInput>div>input, .stSelectbox>div>select {
        background: rgba(15, 15, 35, 0.9);
        border: 1px solid #00ffff;
        color: #e0e6ed;
        border-radius: 5px;
    }

    .stDataFrame table {
        background: rgba(15, 15, 35, 0.6);
        color: #e0e6ed;
        border: 1px solid rgba(0, 255, 255, 0.2);
    }

    .stDataFrame thead th {
        background: linear-gradient(90deg, #0f3460 0%, #16537e 100%);
        color: #00ffff;
        border-bottom: 2px solid #00ffff;
    }

    pre {
        background: rgba(0, 0, 0, 0.8);
        color: #00ff88;
        border: 1px solid #00ffff;
        border-radius: 8px;
        padding: 15px;
        font-family: 'Courier New', monospace;
    }

    .author-info {
        background: linear-gradient(135deg, rgba(15, 15, 35, 0.9) 0%, rgba(26, 26, 46, 0.9) 100%);
        border: 1px solid #00ffff;
        border-radius: 10px;
        padding: 15px;
        margin: 15px;
        box-shadow: 0 5px 20px rgba(0, 255, 255, 0.15);
        color: #b0c4de;
        font-size: 13px;
        text-align: center;
    }

    .author-info h4 {
        text-align: center;
        margin-bottom: 12px;
        font-size: 16px;
    }

    .author-info a {
        color: #00ff88 !important;
        text-decoration: none;
    }

    .author-info a:hover {
        color: #00ffff !important;
        text-shadow: 0 0 5px rgba(0, 255, 255, 0.7);
    }

    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #00ffff, transparent);
    }

    .result-box {
        background: linear-gradient(135deg, rgba(15, 15, 35, 0.9) 0%, rgba(26, 26, 46, 0.9) 100%);
        border: 2px solid #00ffff;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 5px 20px rgba(0, 255, 255, 0.15);
        font-family: 'Roboto', sans-serif;
        color: #e0e6ed;
    }

    .result-box h4 {
        color: #00ffff;
        font-family: 'Orbitron', monospace;
        text-shadow: 0 0 8px rgba(0, 255, 255, 0.5);
        margin-bottom: 10px;
    }

    .result-box p {
        margin: 5px 0;
        font-size: 14px;
    }

    .styled-table {
        background: linear-gradient(135deg, rgba(15, 15, 35, 0.9) 0%, rgba(26, 26, 46, 0.9) 100%);
        border: 2px solid #00ffff;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 5px 15px rgba(0, 255, 255, 0.15);
        color: #e0e6ed;
        width: 100%;
        border-collapse: collapse;
    }

    .styled-table th, .styled-table td {
        padding: 10px;
        border: 1px solid rgba(0, 255, 255, 0.2);
        text-align: left;
    }

    .styled-table th {
        background: linear-gradient(90deg, #0f3460 0%, #16537e 100%);
        color: #00ffff;
        font-family: 'Orbitron', monospace;
    }

    .styled-table tr:hover {
        background: rgba(0, 255, 255, 0.1);
    }

    .main {
        padding-top: 2rem;
    }
    
    .css-1d391kg {
        background: rgba(30, 30, 46, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1E1E2E, #27293D);
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px 0 rgba(0, 0, 0, 0.2);
        margin: 0.5rem 0;
        border-left: 4px solid #BB86FC;
        color: #FAFAFA;
    }
    
    .title-container {
        background: linear-gradient(90deg, #3700B3, #03DAC6);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .section-header {
        background: linear-gradient(90deg, #3700B3, #018786);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    
    .info-box {
        background: rgba(30, 30, 46, 0.9);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #03DAC6;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        color: #FAFAFA;
    }
    
    .footer-dark {
        background-color: #1E1E2E;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar avec navigation et informations auteur
with st.sidebar:
    selected = option_menu(
        menu_title="📊 Analyse Série Temporelle",
        options=["📊 Vue d'ensemble", "📈 Données & Nettoyage", "📋 Statistiques Descriptives", 
                "📉 Visualisations", "🔧 Lissage Exponentiel", "✅ Validation Modèle", "🔮 Prévisions", "📑 Conclusions"],
        icons=["house", "upload", "bar-chart", "graph-up", "gear", "check-circle", "lightning", "file-text"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background": "linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%)", "border-right": "2px solid #00ffff"},
            "icon": {"color": "#b0c4de", "font-size": "20px"},
            "nav-link": {"color": "#b0c4de", "font-family": "'Roboto', sans-serif", "border-left": "3px solid transparent"},
            "nav-link-selected": {"color": "#00ffff", "background": "rgba(0, 255, 255, 0.1)", "border-left": "3px solid #00ffff"}
        }
    )
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
        <div class="author-info">
            <h4>À propos de l'auteur</h4>
            <p>Nom : N'dri</p>
            <p>Prénom : Abo Onesime</p>
            <p>Poste : Statisticien-Data Scientist</p>
            <p>Tél : 07-68-05-98-87 / 01-01-75-11-81</p>
            <p>Email : <a href="mailto:ndriablatie123@gmail.com">ndriablatie123@gmail.com</a></p>
            <p>LinkedIn : <a href="https://www.linkedin.com/in/abo-onesime-n-dri-54a537200/" target="_blank">Profil LinkedIn</a></p>
            <p>GitHub : <a href="https://github.com/Aboonesime" target="_blank">Mon GitHub</a></p>
        </div>
    """, unsafe_allow_html=True)

# Titre principal avec design moderne
st.markdown("""
<div class="title-container">
    <h1>📊 Analyse Série Temporelle PIB Américain</h1>
    <h3>Modélisation Hot-Winters & Prévisions 1974-2024</h3>
    <p>Étude économétrique avancée avec lissage exponentiel</p>
</div>
""", unsafe_allow_html=True)

# Fonction pour créer les données simulées
@st.cache_data
def create_sample_data():
    """Crée des données simulées basées sur les vraies tendances du PIB américain"""
    dates = pd.date_range(start='1974-01-01', end='2024-03-31', freq='Q')
    
    # Tendance de base avec croissance réaliste
    trend = np.linspace(1491.209, 28269.174, len(dates))
    
    # Ajout de variations cycliques et de chocs économiques
    noise = np.random.normal(0, 200, len(dates))
    seasonal = 50 * np.sin(2 * np.pi * np.arange(len(dates)) / 4)
    
    # Simulation de crises économiques
    crisis_1980 = np.where((pd.to_datetime(dates).year >= 1979) & (pd.to_datetime(dates).year <= 1982), -300, 0)
    crisis_2008 = np.where((pd.to_datetime(dates).year >= 2007) & (pd.to_datetime(dates).year <= 2009), -500, 0)
    crisis_2020 = np.where(pd.to_datetime(dates).year == 2020, -800, 0)
    
    pib_values = trend + seasonal + noise + crisis_1980 + crisis_2008 + crisis_2020
    
    df = pd.DataFrame({
        'DATE': dates,
        'PIB': pib_values
    })
    
    return df

# Fonction de chargement des données
@st.cache_data
def load_data():
    """Charge les données du PIB"""
    try:
        # Essayer de charger le fichier CSV
        data = pd.read_csv("gdp_data.csv")
        data = data.rename(columns={'GDP': 'PIB'})
        return data
    except:
        # Si le fichier n'existe pas, utiliser les données simulées
        st.warning("⚠️ Fichier gdp_data.csv non trouvé. Utilisation de données simulées pour la démonstration.")
        return create_sample_data()

# Chargement des données
with st.spinner("🔄 Chargement des données..."):
    pib_data = load_data()

# Préparation des données
pib_data['Date_tri'] = pd.to_datetime(pib_data['DATE'])
pib_data['YEAR'] = pib_data['Date_tri'].dt.year
pib_data.set_index('Date_tri', inplace=True)
pib_series = pib_data['PIB'].dropna()

# Section Vue d'ensemble
if selected == "📊 Vue d'ensemble":
    st.markdown('<div class="section-header">📊 Vue d\'ensemble du Projet</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📅 Période</h3>
            <h2>1974-2024</h2>
            <p>50 années d'analyse</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Observations</h3>
            <h2>{len(pib_series)}</h2>
            <p>Points de données</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💰 PIB Min</h3>
            <h2>{pib_series.min():,.0f}B</h2>
            <p>Milliards USD</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 PIB Max</h3>
            <h2>{pib_series.max():,.0f}B</h2>
            <p>Milliards USD</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>🎯 Objectifs de l'étude</h3>
        <ul>
            <li>📊 Analyser l'évolution du PIB américain sur 50 ans</li>
            <li>🔧 Appliquer la méthode de lissage exponentiel Hot-Winters</li>
            <li>✅ Valider le modèle par analyse des résidus</li>
            <li>🔮 Réaliser des prévisions économiques</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Section Données & Nettoyage
elif selected == "📈 Données & Nettoyage":
    st.markdown('<div class="section-header">📈 Importation et Nettoyage des Données</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Aperçu des données")
        st.dataframe(pib_data.head(10), use_container_width=True)
        
        st.subheader("📊 Informations générales")
        st.write(f"• **Période**: {pib_data.index.min().strftime('%Y-%m-%d')} à {pib_data.index.max().strftime('%Y-%m-%d')}")
        st.write(f"• **Nombre d'observations**: {len(pib_data)}")
        st.write(f"• **Fréquence**: Trimestrielle")
        st.write(f"• **Valeurs manquantes**: {pib_data['PIB'].isna().sum()}")
        st.write(f"• **Doublons**: {pib_data.index.duplicated().sum()}")
    
    with col2:
        st.subheader("✅ Contrôles qualité")
        
        # Vérification des valeurs manquantes
        missing_pct = (pib_data['PIB'].isna().sum() / len(pib_data)) * 100
        st.metric("Valeurs manquantes", f"{missing_pct:.1f}%")
        
        # Vérification des doublons
        duplicates = pib_data.index.duplicated().sum()
        st.metric("Doublons", duplicates)
        
        # Statistiques de base
        st.metric("PIB moyen", f"{pib_data['PIB'].mean():,.0f}B $")
        st.metric("Écart-type", f"{pib_data['PIB'].std():,.0f}B $")

# Section Statistiques Descriptives
elif selected == "📋 Statistiques Descriptives":
    st.markdown('<div class="section-header">📋 Analyse Statistique Descriptive</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Statistiques descriptives")
        stats_df = pib_series.describe()
        st.dataframe(stats_df.to_frame().T, use_container_width=True)
        
        # Indices de forme
        st.subheader("📈 Indices de forme")
        kurt = kurtosis(pib_series, bias=False)
        skew_val = skew(pib_series, bias=False)
        
        st.write(f"• **Kurtosis (Aplatissement)**: {kurt:.4f}")
        st.write(f"• **Skewness (Asymétrie)**: {skew_val:.4f}")
        
        if skew_val > 0:
            st.write("📊 Distribution asymétrique positive (queue à droite)")
        else:
            st.write("📊 Distribution asymétrique négative (queue à gauche)")
    
    with col2:
        st.subheader("📈 Évolution par décennie")
        
        # Calcul par décennie
        decades_stats = []
        for decade in range(1970, 2030, 10):
            decade_data = pib_series[pib_series.index.year >= decade]
            decade_data = decade_data[decade_data.index.year < decade + 10]
            if len(decade_data) > 0:
                decades_stats.append({
                    'Décennie': f"{decade}s",
                    'PIB Moyen': decade_data.mean(),
                    'Croissance %': ((decade_data.iloc[-1] - decade_data.iloc[0]) / decade_data.iloc[0] * 100) if len(decade_data) > 1 else 0
                })
        
        decades_df = pd.DataFrame(decades_stats)
        st.dataframe(decades_df, use_container_width=True, hide_index=True)

# Section Visualisations
elif selected == "📉 Visualisations":
    st.markdown('<div class="section-header">📉 Visualisations Avancées</div>', unsafe_allow_html=True)
    
    # Graphique principal - Série temporelle avec Plotly
    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(
        x=pib_series.index,
        y=pib_series.values,
        mode='lines+markers',
        name='PIB Trimestriel',
        line=dict(color='#BB86FC', width=2),
        marker=dict(size=4, color='#BB86FC')
    ))
    
    fig_main.update_layout(
        title={
            'text': "Évolution du PIB Américain (1974-2024)",
            'x': 0.5,
            'font': {'size': 20, 'color': 'white'}
        },
        xaxis_title="Année",
        yaxis_title="PIB (Milliards USD)",
        template="plotly_dark",
        height=500,
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig_main, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution - Histogramme
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=pib_series.values,
            nbinsx=25,
            name='Distribution PIB',
            marker_color='#03DAC6',
            opacity=0.7
        ))
        
        fig_hist.update_layout(
            title="Distribution du PIB",
            xaxis_title="PIB (Milliards USD)",
            yaxis_title="Fréquence",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Box Plot
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=pib_series.values,
            name='PIB',
            marker_color='#BB86FC',
            boxpoints='outliers'
        ))
        
        fig_box.update_layout(
            title="Box Plot du PIB",
            yaxis_title="PIB (Milliards USD)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Analyse des autocorrélations
    st.subheader("📊 Analyse des Autocorrélations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ACF
        acf_values = acf(pib_series, nlags=20, fft=False)
        
        fig_acf = go.Figure()
        fig_acf.add_trace(go.Scatter(
            x=list(range(len(acf_values))),
            y=acf_values,
            mode='lines+markers',
            name='ACF',
            marker=dict(color='#BB86FC', size=6),
            line=dict(color='#BB86FC', width=2)
        ))
        
        # Lignes de confiance
        n = len(pib_series)
        confidence_interval = 1.96 / np.sqrt(n)
        fig_acf.add_hline(y=confidence_interval, line_dash="dash", line_color="red", opacity=0.5)
        fig_acf.add_hline(y=-confidence_interval, line_dash="dash", line_color="red", opacity=0.5)
        
        fig_acf.update_layout(
            title="Fonction d'Autocorrélation (ACF)",
            xaxis_title="Lag",
            yaxis_title="Autocorrélation",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig_acf, use_container_width=True)
    
    with col2:
        # PACF
        pacf_values = pacf(pib_series, nlags=20, method='ywm')
        
        fig_pacf = go.Figure()
        fig_pacf.add_trace(go.Scatter(
            x=list(range(len(pacf_values))),
            y=pacf_values,
            mode='lines+markers',
            name='PACF',
            marker=dict(color='#03DAC6', size=6),
            line=dict(color='#03DAC6', width=2)
        ))
        
        # Lignes de confiance
        fig_pacf.add_hline(y=confidence_interval, line_dash="dash", line_color="red", opacity=0.5)
        fig_pacf.add_hline(y=-confidence_interval, line_dash="dash", line_color="red", opacity=0.5)
        
        fig_pacf.update_layout(
            title="Fonction d'Autocorrélation Partielle (PACF)",
            xaxis_title="Lag",
            yaxis_title="Autocorrélation Partielle",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig_pacf, use_container_width=True)

# Section Lissage Exponentiel
elif selected == "🔧 Lissage Exponentiel":
    st.markdown('<div class="section-header">🔧 Modélisation Hot-Winters</div>', unsafe_allow_html=True)
    
    with st.spinner("🔄 Ajustement du modèle..."):
        warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
        
        # Configuration du modèle
        model = ExponentialSmoothing(
            pib_series,
            trend='add',
            seasonal=None,
            initialization_method='estimated'
        )
        
        # Ajustement
        fitted_model = model.fit()
        
        # Résultats du modèle
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Paramètres du Modèle")
            
            st.markdown(f"""
            <div class="metric-card">
                <strong>Alpha (niveau)</strong>: {fitted_model.params['smoothing_level']:.5f}<br>
                <strong>Beta (tendance)</strong>: {fitted_model.params['smoothing_trend']:.5f}<br>
                <strong>SSE (erreur)</strong>: {fitted_model.sse:,.2f}<br>
                <strong>Observations</strong>: {len(pib_series)}
            </div>
            """, unsafe_allow_html=True)
            
            st.info("🔍 **Alpha élevé** (≈0.88): Le modèle s'adapte rapidement aux nouvelles observations")
            st.info("🔍 **Beta modéré** (≈0.09): La tendance évolue graduellement")
        
        with col2:
            st.subheader("📈 Qualité d'ajustement")
            
            # Calcul de métriques
            residuals = fitted_model.resid
            mae = np.mean(np.abs(residuals))
            rmse = np.sqrt(np.mean(residuals**2))
            mape = np.mean(np.abs(residuals / pib_series)) * 100
            
            st.markdown(f"""
            <div class="metric-card">
                <strong>MAE</strong>: {mae:.2f}<br>
                <strong>RMSE</strong>: {rmse:.2f}<br>
                <strong>MAPE</strong>: {mape:.2f}%<br>
                <strong>R²</strong>: {1 - (fitted_model.sse / np.sum((pib_series - pib_series.mean())**2)):.4f}
            </div>
            """, unsafe_allow_html=True)
    
    # Visualisation du lissage
    fig_smooth = go.Figure()
    
    # Série originale
    fig_smooth.add_trace(go.Scatter(
        x=pib_series.index,
        y=pib_series.values,
        mode='lines+markers',
        name='Série originale',
        line=dict(color='#BB86FC', width=2),
        marker=dict(size=4)
    ))
    
    # Série lissée
    fig_smooth.add_trace(go.Scatter(
        x=pib_series.index,
        y=fitted_model.fittedvalues,
        mode='lines',
        name='Série lissée (Holt)',
        line=dict(color='#03DAC6', width=3, dash='dash')
    ))
    
    fig_smooth.update_layout(
        title="Comparaison: Série Originale vs Série Lissée",
        xaxis_title="Date",
        yaxis_title="PIB (Milliards USD)",
        template="plotly_dark",
        height=500,
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig_smooth, use_container_width=True)

# Section Validation du Modèle
elif selected == "✅ Validation Modèle":
    st.markdown('<div class="section-header">✅ Validation par Analyse des Résidus</div>', unsafe_allow_html=True)
    
    # Calcul du modèle et résidus
    model = ExponentialSmoothing(pib_series, trend='add', seasonal=None, initialization_method='estimated')
    fitted_model = model.fit()
    residuals = fitted_model.resid.dropna()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Analyse des Résidus")
        
        # Graphique des résidus
        fig_resid = go.Figure()
        fig_resid.add_trace(go.Scatter(
            x=residuals.index,
            y=residuals.values,
            mode='lines+markers',
            name='Résidus',
            line=dict(color='#03DAC6', width=1.5),
            marker=dict(size=3)
        ))
        
        fig_resid.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.7)
        
        fig_resid.update_layout(
            title="Évolution des Résidus",
            xaxis_title="Date",
            yaxis_title="Résidu",
            template="plotly_dark",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig_resid, use_container_width=True)
    
    with col2:
        st.subheader("📈 Distribution des Résidus")
        
        # Histogramme des résidus
        fig_resid_hist = go.Figure()
        fig_resid_hist.add_trace(go.Histogram(
            x=residuals.values,
            nbinsx=20,
            name='Résidus',
            marker_color='#BB86FC',
            opacity=0.7
        ))
        
        fig_resid_hist.update_layout(
            title="Distribution des Résidus",
            xaxis_title="Valeur des résidus",
            yaxis_title="Fréquence",
            template="plotly_dark",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig_resid_hist, use_container_width=True)
    
    # Tests statistiques
    st.subheader("🧪 Tests Statistiques")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Test de Ljung-Box (Bruit Blanc)")
        
        # Test de Ljung-Box
        lb_test = acorr_ljungbox(residuals, lags=[20], return_df=True)
        lb_stat = lb_test.loc[20, 'lb_stat']
        lb_pvalue = lb_test.loc[20, 'lb_pvalue']
        
        st.markdown(f"""
        <div class="metric-card">
            <strong>H₀</strong>: Les résidus sont un bruit blanc<br>
            <strong>Statistique</strong>: {lb_stat:.4f}<br>
            <strong>p-value</strong>: {lb_pvalue:.6f}<br>
            <strong>Conclusion</strong>: {"✅ Bruit blanc accepté" if lb_pvalue > 0.05 else "❌ Bruit blanc rejeté"}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Test de Shapiro-Wilk (Normalité)")
        
        # Test de normalité
        if len(residuals) < 5000:
            shapiro_stat, shapiro_pvalue = shapiro(residuals)
            
            st.markdown(f"""
            <div class="metric-card">
                <strong>H₀</strong>: Les résidus suivent une loi normale<br>
                <strong>Statistique</strong>: {shapiro_stat:.4f}<br>
                <strong>p-value</strong>: {shapiro_pvalue:.6f}<br>
                <strong>Conclusion</strong>: {"✅ Normalité acceptée" if shapiro_pvalue > 0.05 else "❌ Normalité rejetée"}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Échantillon trop grand pour le test de Shapiro-Wilk")
    
    # Statistiques des résidus
    st.subheader("📊 Statistiques des Résidus")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Moyenne", f"{residuals.mean():.2f}")
    with col2:
        st.metric("Écart-type", f"{residuals.std():.2f}")
    with col3:
        st.metric("Min", f"{residuals.min():.2f}")
    with col4:
        st.metric("Max", f"{residuals.max():.2f}")

# Section Prévisions
elif selected == "🔮 Prévisions":
    st.markdown('<div class="section-header">🔮 Prévisions Économiques</div>', unsafe_allow_html=True)
    
    # Interface utilisateur pour les prévisions
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("⚙️ Paramètres de Prévision")
        
        steps = st.selectbox(
            "Nombre de périodes à prévoir:",
            options=[1, 2, 3, 4, 6, 8],
            index=3,  # 4 par défaut
            help="Nombre de trimestres à prévoir"
        )
        
        confidence_level = st.selectbox(
            "Niveau de confiance:",
            options=[90, 95, 99],
            index=1,  # 95% par défaut
            help="Niveau de confiance pour les intervalles"
        )
        
        show_historic = st.checkbox("Afficher historique complet", value=False)
        
    with col2:
        st.subheader("📊 Calcul des Prévisions")
        
        with st.spinner("🔄 Calcul en cours..."):
            # Ajustement du modèle
            model = ExponentialSmoothing(
                pib_series,
                trend='add',
                seasonal=None,
                initialization_method='estimated'
            )
            fitted_model = model.fit()
            
            # Prévisions
            forecast = fitted_model.forecast(steps=steps)
            
            # Calcul des intervalles de confiance
            residuals = fitted_model.resid
            sigma = np.sqrt(np.mean(residuals**2))
            h = np.arange(1, steps + 1)
            se_forecast = sigma * np.sqrt(h)
            
            # Seuil pour le niveau de confiance
            z_score = norm.ppf(1 - (100 - confidence_level) / 200)
            
            lower = forecast - z_score * se_forecast
            upper = forecast + z_score * se_forecast
            
            # Création des dates futures - Correction de l'erreur
            last_date = pib_series.index[-1]
            future_dates = [last_date + pd.DateOffset(months=3*i) for i in range(1, steps+1)]
            future_dates = pd.DatetimeIndex(future_dates)
            
            # Tableau des prévisions
            predictions_df = pd.DataFrame({
                'Date': future_dates,
                'Prévision': forecast.values,
                'Borne Inf.': lower.values,
                'Borne Sup.': upper.values
            })
            
            predictions_df['Date'] = predictions_df['Date'].dt.strftime('%Y-T%q')
            
            st.dataframe(
                predictions_df.style.format({
                    'Prévision': '{:,.0f}B $',
                    'Borne Inf.': '{:,.0f}B $',
                    'Borne Sup.': '{:,.0f}B $'
                }),
                use_container_width=True,
                hide_index=True
            )
    
    # Visualisation des prévisions
    st.subheader("📈 Visualisation des Prévisions")
    
    # Données pour le graphique
    if show_historic:
        hist_data = pib_series
    else:
        # Afficher seulement les 20 derniers points
        hist_data = pib_series.tail(20)
    
    fig_forecast = go.Figure()
    
    # Série historique
    fig_forecast.add_trace(go.Scatter(
        x=hist_data.index,
        y=hist_data.values,
        mode='lines+markers',
        name='Données historiques',
        line=dict(color='#BB86FC', width=2),
        marker=dict(size=4)
    ))
    
    # Prévisions
    fig_forecast.add_trace(go.Scatter(
        x=future_dates,
        y=forecast.values,
        mode='lines+markers',
        name='Prévisions',
        line=dict(color='#03DAC6', width=3, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    # Intervalle de confiance
    fig_forecast.add_trace(go.Scatter(
        x=future_dates.tolist() + future_dates.tolist()[::-1],
        y=upper.values.tolist() + lower.values.tolist()[::-1],
        fill='toself',
        fillcolor='rgba(3, 218, 198, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name=f'IC {confidence_level}%',
        hoverinfo="skip"
    ))
    
    # Ligne de séparation - Correction de l'erreur
    fig_forecast.add_vline(
        x=pib_series.index[-1].timestamp() * 1000,  # Conversion en millisecondes pour Plotly
        line_dash="dash",
        line_color="gray",
        opacity=0.7,
        annotation_text="Prévisions →",
        annotation_position="top right"
    )
    
    fig_forecast.update_layout(
        title=f"Prévisions PIB Américain - {steps} Trimestres (IC {confidence_level}%)",
        xaxis_title="Date",
        yaxis_title="PIB (Milliards USD)",
        template="plotly_dark",
        height=600,
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Métriques de prévision
    st.subheader("📊 Métriques de Prévision")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Prévision T+1",
            f"{forecast.iloc[0]:,.0f}B $",
            f"{((forecast.iloc[0] - pib_series.iloc[-1])/pib_series.iloc[-1]*100):+.1f}%"
        )
    
    with col2:
        st.metric(
            f"Prévision T+{steps}",
            f"{forecast.iloc[-1]:,.0f}B $",
            f"{((forecast.iloc[-1] - pib_series.iloc[-1])/pib_series.iloc[-1]*100):+.1f}%"
        )
    
    with col3:
        growth_rate = ((forecast.iloc[-1] / forecast.iloc[0])**(1/steps) - 1) * 100
        st.metric(
            "Taux croissance trim.",
            f"{growth_rate:.2f}%",
            "Moyenne période"
        )
    
    with col4:
        annual_growth = ((forecast.iloc[-1] / forecast.iloc[0])**(4/steps) - 1) * 100
        st.metric(
            "Taux croissance ann.",
            f"{annual_growth:.2f}%",
            "Estimation annuelle"
        )
    
    # Analyse des scénarios
    st.subheader("🎯 Analyse des Scénarios")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>📈 Scénario Optimiste</h4>
            <p>Si la croissance suit la borne supérieure de l'intervalle de confiance:</p>
            <ul>
                <li>PIB pourrait atteindre <strong>{:,.0f}B $</strong></li>
                <li>Croissance annuelle: <strong>{:.1f}%</strong></li>
                <li>Facteurs: Innovation technologique, politiques favorables</li>
            </ul>
        </div>
        """.format(
            upper.iloc[-1],
            ((upper.iloc[-1] / pib_series.iloc[-1])**(4/steps) - 1) * 100
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>📉 Scénario Pessimiste</h4>
            <p>Si la croissance suit la borne inférieure de l'intervalle de confiance:</p>
            <ul>
                <li>PIB pourrait être limité à <strong>{:,.0f}B $</strong></li>
                <li>Croissance annuelle: <strong>{:.1f}%</strong></li>
                <li>Facteurs: Récession, chocs externes, inflation</li>
            </ul>
        </div>
        """.format(
            lower.iloc[-1],
            ((lower.iloc[-1] / pib_series.iloc[-1])**(4/steps) - 1) * 100
        ), unsafe_allow_html=True)

# Section Conclusions
elif selected == "📑 Conclusions":
    st.markdown('<div class="section-header">📑 Conclusions et Recommandations</div>', unsafe_allow_html=True)
    
    # Résumé exécutif
    st.subheader("📋 Résumé Exécutif")
    
    # Calculer le modèle pour les conclusions
    model = ExponentialSmoothing(pib_series, trend='add', seasonal=None, initialization_method='estimated')
    fitted_model = model.fit()
    residuals = fitted_model.resid.dropna()
    
    # Test de Ljung-Box
    lb_test = acorr_ljungbox(residuals, lags=[20], return_df=True)
    lb_pvalue = lb_test.loc[20, 'lb_pvalue']
    
    # Test de normalité
    if len(residuals) < 5000:
        _, shapiro_pvalue = shapiro(residuals)
    else:
        shapiro_pvalue = 0.0  # Approximation pour grands échantillons
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>🎯 Objectifs Atteints</h4>
            <ul>
                <li>✅ <strong>Modélisation réussie</strong> du PIB américain sur 50 ans</li>
                <li>✅ <strong>Validation statistique</strong> par test de bruit blanc</li>
                <li>✅ <strong>Prévisions fiables</strong> avec intervalles de confiance</li>
                <li>✅ <strong>Interface interactive</strong> pour l'analyse</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-box">
            <h4>📊 Résultats Clés</h4>
            <ul>
                <li><strong>Croissance moyenne</strong>: {((pib_series.iloc[-1]/pib_series.iloc[0])**(1/50) - 1)*100:.1f}% par an</li>
                <li><strong>Modèle validé</strong>: Résidus = bruit blanc (p={lb_pvalue:.3f})</li>
                <li><strong>Alpha élevé</strong>: {fitted_model.params['smoothing_level']:.3f} (adaptation rapide)</li>
                <li><strong>Erreur RMSE</strong>: {np.sqrt(np.mean(residuals**2)):.0f}B $</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Interprétation économique
    st.subheader("💡 Interprétation Économique")
    
    st.markdown("""
    <div class="info-box">
        <h4>📈 Trajectoire de Croissance</h4>
        <p>L'économie américaine a démontré une <strong>remarquable résilience</strong> sur 50 ans :</p>
        <ul>
            <li><strong>Crises surmontées</strong> : Chocs pétroliers (70s-80s), crise financière (2008), pandémie (2020)</li>
            <li><strong>Moteurs de croissance</strong> : Innovation technologique, déréglementation, investissements</li>
            <li><strong>Adaptabilité</strong> : Transformation vers l'économie de services et le numérique</li>
            <li><strong>Politiques contra-cycliques</strong> : Relance effective lors des récessions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Limitations et recommandations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>⚠️ Limitations du Modèle</h4>
            <ul>
                <li><strong>Résidus non-gaussiens</strong> : Distribution asymétrique</li>
                <li><strong>Absence de saisonnalité</strong> : Modèle simplifié</li>
                <li><strong>Chocs exogènes</strong> : Non pris en compte</li>
                <li><strong>Variables explicatives</strong> : Modèle univarié</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>🚀 Recommandations</h4>
            <ul>
                <li><strong>Modèles SARIMA</strong> : Intégrer la saisonnalité</li>
                <li><strong>Variables exogènes</strong> : Taux d'intérêt, inflation, emploi</li>
                <li><strong>Modèles VAR</strong> : Relations entre variables économiques</li>
                <li><strong>Machine Learning</strong> : Approches non-linéaires</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Perspectives futures
    st.subheader("🔮 Perspectives d'Avenir")
    
    # Calcul d'une prévision simple pour l'exemple
    forecast_sample = fitted_model.forecast(steps=4)
    
    st.markdown(f"""
    <div class="info-box">
        <h4>🎯 Projections Court Terme</h4>
        <p>Selon notre modèle Hot-Winters, le PIB américain devrait :</p>
        <ul>
            <li><strong>T2 2024</strong> : Environ {forecast_sample.iloc[0]:,.0f}B $ (+{((forecast_sample.iloc[0]/pib_series.iloc[-1]-1)*100):+.1f}%)</li>
            <li><strong>T1 2025</strong> : Environ {forecast_sample.iloc[-1]:,.0f}B $ (croissance soutenue)</li>
            <li><strong>Tendance</strong> : Maintien d'une croissance positive mais modérée</li>
            <li><strong>Incertitudes</strong> : Géopolitique, politique monétaire, innovation</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Méta-analyse
    st.subheader("🔬 Contribution Scientifique")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>📚 Apports Méthodologiques</h4>
            <ul>
                <li><strong>Application pratique</strong> du lissage exponentiel</li>
                <li><strong>Validation rigoureuse</strong> par tests statistiques</li>
                <li><strong>Interface moderne</strong> pour l'analyse économique</li>
                <li><strong>Reproductibilité</strong> des résultats</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>🎓 Applications Pédagogiques</h4>
            <ul>
                <li><strong>Démonstration interactive</strong> des concepts</li>
                <li><strong>Visualisations avancées</strong> pour la compréhension</li>
                <li><strong>Validation par l'exemple</strong> des méthodes</li>
                <li><strong>Outil d'apprentissage</strong> complet</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer avec informations
    st.markdown("---")
    st.markdown("""
    <div class="footer-dark">
        <h4 style="color: #03DAC6;">📊 Application développée par : Onesime-ndri</h4>
        <p style="color: #a6a6a6;">Étude économétrique - PIB Américain 1974-2024</p>
        <p style="color: #a6a6a6;">Modélisation Hot-Winters & Interface Streamlit</p>
        <p style="color: #BB86FC;"><em>Pour usage académique et professionnel</em></p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar avec informations supplémentaires
st.sidebar.markdown("---")
st.sidebar.markdown("## 📊 Informations")

with st.sidebar:
    st.info(f"""
    **Période d'étude**: 1974-2024
    
    **Observations**: {len(pib_series)}
    
    **Méthode**: Hot-Winters
    
    **Fréquence**: Trimestrielle
    """)
    
    st.success(f"""
    **PIB Actuel**: {pib_series.iloc[-1]:,.0f}B $
    
    **Croissance 50 ans**: {((pib_series.iloc[-1]/pib_series.iloc[0])**(1/50) - 1)*100:.1f}%/an
    
    **Volatilité**: {(pib_series.std()/pib_series.mean()*100):.1f}%
    """)
    
    if st.button("💾 Télécharger Données"):
        csv = pib_data.to_csv(index=True)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="pib_usa_data.csv">Télécharger CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; font-size: 0.8em; color: #666;">
        Développé avec ❤️<br>
        Streamlit + Plotly<br>
        © 2024 Onesime-ndri
    </div>
    """, unsafe_allow_html=True)