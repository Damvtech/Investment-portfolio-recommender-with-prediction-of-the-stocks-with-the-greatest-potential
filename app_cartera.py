import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Fechas globales
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

# Título y disclaimer
st.title("💼 Asesor de Cartera de Inversión")
st.markdown("*Esta simulación se basa en datos históricos y no garantiza rentabilidades futuras. Invierte con responsabilidad.*")

@st.cache_data
def choose_tickers():
    df = pd.read_csv('top_28_growth_stocks.csv')
    tickers = df['Ticker'].tolist()
    names = df['name'].tolist()

    tickers_validos = []
    names_validos = []
    for ticker, name in zip(tickers, names):
        try:
            data = yf.Ticker(ticker).history(period="1d")
            if not data.empty:
                tickers_validos.append(ticker)
                names_validos.append(name)
        except:
            pass
    return tickers_validos, names_validos

@st.cache_data
def cargar_datos():
    tickers, names = choose_tickers()
    symbols = dict(zip(names, tickers))
    data = pd.DataFrame()
    for company, symbol in symbols.items():
        df = yf.download(symbol, start=start_date, end=end_date)
        if not df.empty and 'Close' in df.columns:
            data[company] = df['Close']
    return data.dropna(), tickers, names

def optimizar_cartera(mean_returns, cov_matrix, perfil):
    def annualized_return(w):
        return np.sum(mean_returns * w) * 252

    def annualized_volatility(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) * np.sqrt(252)

    objetivos = {
        "Riesgo mínimo": lambda w: annualized_volatility(w),
        "Riesgo bajo": lambda w: 0.75*annualized_volatility(w)-0.25*annualized_return(w),
        "Riesgo medio": lambda w: 0.50*annualized_volatility(w)-0.50*annualized_return(w),
        "Riesgo alto": lambda w: 0.25*annualized_volatility(w)-0.75*annualized_return(w),
        "Rentabilidad máxima": lambda w: -annualized_return(w)
    }

    n = len(mean_returns)
    w0 = np.ones(n) / n
    bounds = [(0, 1) for _ in range(n)]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    result = minimize(objetivos[perfil], w0, bounds=bounds, constraints=constraints)
    return result.x

def determinar_perfil(respuestas):
    total = sum(respuestas)
    if total <= 8:
        return "Riesgo mínimo"
    elif total <= 12:
        return "Riesgo bajo"
    elif total <= 17:
        return "Riesgo medio"
    elif total <= 21:
        return "Riesgo alto"
    else:
        return "Rentabilidad máxima"

# 🎛️ Barra lateral con preguntas
with st.sidebar:
    st.header("🧠 Configura tu perfil")
    opciones = {
        "Nivel de seguridad deseado": ["Máxima seguridad", "Alta seguridad", "Equilibrado", "Alta rentabilidad", "Máxima rentabilidad"],
        "Diversificación deseada": ["Muy conservadora", "Conservadora", "Equilibrada", "Agresiva", "Muy agresiva"],
        "Expectativa de rentabilidad": ["Muy baja", "Moderada-baja", "Media", "Alta", "Muy alta"],
        "Nivel máximo de pérdida aceptable": ["0%", "Hasta 5%", "Hasta 10%", "Hasta 20%", "Más de 20%"],
        "Horizonte temporal": ["< 1 año", "1-3 años", "3-5 años", "5-10 años", ">10 años"]
    }

    respuestas = []
    for pregunta, valores in opciones.items():
        eleccion = st.selectbox(pregunta, valores, key=pregunta)
        respuestas.append(valores.index(eleccion) + 1)

    ejecutar = st.button("📊 Generar cartera")

# Solo si se pulsa el botón
if ejecutar:
    st.subheader("📊 Resultados de la simulación")
    perfil = determinar_perfil(respuestas)
    st.markdown(f"**Perfil de riesgo detectado:** `{perfil}`")

    with st.spinner("🔍 Calculando cartera óptima..."):
        data, tickers, names = cargar_datos()
        returns = data.pct_change().dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        pesos = optimizar_cartera(mean_returns, cov_matrix, perfil)

        # Mostrar tabla de pesos
        cartera = {data.columns[i]: round(pesos[i]*100, 2) for i in range(len(data.columns)) if pesos[i] > 0.001}
        ordenada = dict(sorted(cartera.items(), key=lambda x: x[1], reverse=True))
        st.write("### 📌 Composición de la cartera:")
        st.dataframe(pd.DataFrame(ordenada.items(), columns=["Empresa", "Porcentaje"]))

        # Evolución
        inversion_inicial = 1000
        cartera_retornos = (data * pesos).sum(axis=1)
        cartera_valores = cartera_retornos / cartera_retornos.iloc[0] * inversion_inicial

        st.line_chart(cartera_valores, height=300)

        # Estadísticas
        rolling_max = cartera_valores.cummax()
        drawdowns = (cartera_valores - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        drawdown_start = drawdowns.idxmin()
        try:
            recovery = cartera_valores.loc[drawdown_start:]
            drawdown_end = recovery[recovery >= recovery.iloc[0]].index[0]
        except:
            drawdown_end = cartera_valores.index[-1]

        valor_final = cartera_valores.iloc[-1]
        ganancia_pct = (valor_final - inversion_inicial) / inversion_inicial * 100

        st.markdown(f"**📈 Rentabilidad final estimada:** `{valor_final:.2f} €`")
        st.markdown(f"**📉 Máximo drawdown:** `{max_drawdown*100:.2f}%` entre `{drawdown_start.date()}` y `{drawdown_end.date()}`")
        st.markdown(f"**🔎 Rentabilidad acumulada:** `{ganancia_pct:.2f}%`")

