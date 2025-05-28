import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.optimize import minimize
from datetime import datetime, timedelta

st.title("💼 Asesor de Cartera de Inversión")
st.markdown("*Esta simulación se basa en datos históricos y no garantiza rentabilidades futuras. Invierte con responsabilidad.*")

# --- Preguntas de perfil de riesgo ---
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
    cantidad_invertir = st.number_input("💶 Cantidad a invertir (€)", min_value=100, value=1000, step=100)
    ejecutar = st.button("📊 Generar cartera")

# --- Resto del código (ejemplo de carga de datos y lógica principal) ---
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

@st.cache_data
def choose_tickers():
    df = pd.read_csv("historical_prices_long_with_names.csv")
    if df.empty:
        st.error("❌ No se encontraron datos de acciones. Por favor, verifica la fuente de datos.")
        return [], []
    tickers = df['Ticker'].tolist()
    names = df['name'].tolist()
    return tickers, names

@st.cache_data
def cargar_datos():
    df_prices = pd.read_csv("historical_prices_long_with_names.csv", parse_dates=["Date"])
    df_prices = df_prices.drop_duplicates(subset=["Date", "Ticker"])
    
    # Pivotar con tickers como columnas
    df_pivot = df_prices.pivot(index="Date", columns="Ticker", values="Close")

    # Eliminar columnas con muchos NaN
    df_pivot = df_pivot.dropna(axis=1, thresh=len(df_pivot)*0.5)

    # Seleccionar los tickers con más datos
    top_tickers = df_pivot.count().sort_values(ascending=False).head(50).index
    df_pivot = df_pivot[top_tickers]

    # Crear un diccionario Ticker -> Nombre
    name_map = df_prices.drop_duplicates(subset="Ticker").set_index("Ticker")["name"].to_dict()
    top_names = [name_map[t] for t in top_tickers]

    return df_pivot.dropna(), list(top_tickers), top_names


def optimizar_cartera(mean_returns, cov_matrix, perfil, volatilities, threshold=0.6):
    n = len(mean_returns)
    if n == 0:
        st.error("❌ No hay suficientes datos de retorno para calcular una cartera óptima.")
        return np.array([])
    def annualized_return(w):
        return np.sum(mean_returns * w) * 252
    def annualized_volatility(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) * np.sqrt(252)
    objetivos = {
        "Riesgo mínimo": lambda w: annualized_volatility(w),
        "Riesgo bajo": lambda w: 0.75*annualized_volatility(w) - 0.25*annualized_return(w),
        "Riesgo medio": lambda w: 0.50*annualized_volatility(w) - 0.50*annualized_return(w),
        "Riesgo alto": lambda w: 0.25*annualized_volatility(w) - 0.75*annualized_return(w),
        "Rentabilidad máxima": lambda w: -annualized_return(w)
    }
    bounds = []
    for i in range(n):
        vol = volatilities[i]
        if vol > threshold:
            if perfil in ["Riesgo mínimo", "Riesgo bajo"]:
                bounds.append((0, 0))
            elif perfil == "Riesgo medio":
                bounds.append((0, 0.1))
            else:
                bounds.append((0, 0.2))
        else:
            bounds.append((0, 1))
    if all(b[1] == 0 for b in bounds):
        st.error("❌ Todos los activos han sido excluidos por ser demasiado volátiles para tu perfil de riesgo.")
        return np.array([])
    w0 = np.ones(n) / n
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    result = minimize(objetivos[perfil], w0, bounds=bounds, constraints=constraints)
    return result.x

if ejecutar:
    st.subheader("📊 Resultados de la simulación")
    perfil = determinar_perfil(respuestas)
    st.markdown(f"**Perfil de riesgo detectado:** `{perfil}`")
    with st.spinner("🔍 Calculando cartera óptima..."):
        data, tickers, names = cargar_datos()
        if data.empty:
            st.warning("No se pudo generar una cartera óptima con los datos disponibles.")
            st.stop()
        returns = data.pct_change().dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        volatilities = returns.std() * np.sqrt(252)
        pesos = optimizar_cartera(mean_returns, cov_matrix, perfil, volatilities)
        if pesos.size == 0:
            st.warning("No se pudo generar una cartera óptima con los datos disponibles.")
            st.stop()
        # Calcular desglose monetario usando los nombres de empresa
        ticker_to_name = {tick: name for tick, name in zip(tickers, names)}
        cartera = {
            ticker_to_name[data.columns[i]]: round(pesos[i] * cantidad_invertir, 2)
            for i in range(len(data.columns)) if pesos[i] > 0.001
        }
        ordenada = dict(sorted(cartera.items(), key=lambda x: x[1], reverse=True))
        st.write("### 📌 Desglose de inversión por acción (€):")
        st.dataframe(pd.DataFrame(ordenada.items(), columns=["Empresa", "Cantidad (€)"]))
        inversion_inicial = cantidad_invertir
        cartera_retornos = (data * pesos).sum(axis=1)
        cartera_valores = cartera_retornos / cartera_retornos.iloc[0] * inversion_inicial
        st.line_chart(cartera_valores, height=300)
        rolling_max = cartera_valores.cummax()
        drawdowns = (cartera_valores - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        drawdown_min_idx = np.argmin(drawdowns)
        peak_idx = np.argmax(cartera_valores.values[:drawdown_min_idx+1])
        drawdown_start = cartera_valores.index[peak_idx]
        drawdown_end = cartera_valores.index[drawdown_min_idx]
        valor_final = cartera_valores.iloc[-1]
        ganancia_pct = (valor_final - inversion_inicial) / inversion_inicial * 100
        st.markdown(f"**📈 Rentabilidad final estimada:** `{valor_final:.2f} €`")
        st.markdown(f"**📉 Máximo drawdown:** `{max_drawdown*100:.2f}%` entre `{drawdown_start.strftime('%Y-%m-%d')}` y `{drawdown_end.strftime('%Y-%m-%d')}`")
        st.markdown(f"**🔎 Rentabilidad acumulada:** `{ganancia_pct:.2f}%`")