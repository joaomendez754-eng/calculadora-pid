import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import tf2ss, cont2discrete

st.set_page_config(page_title="Calculadora PID", layout="wide")

st.title("Calculadora PID")
st.write("Ingresa tu planta y modifica las ganancias para observar la respuesta.")


def leer_coeficientes(texto):
    texto = texto.replace("[", "").replace("]", "").replace(",", " ")
    valores = np.array([float(x) for x in texto.split()])

    if valores.size == 0 or not np.all(np.isfinite(valores)):
        raise ValueError("Ingresa coeficientes numéricos válidos.")

    valores = np.trim_zeros(valores, "f")
    return valores


def polinomio(coeficientes):
    partes = []
    grado = len(coeficientes) - 1

    for indice, valor in enumerate(coeficientes):
        if valor == 0:
            continue

        potencia = grado - indice
        magnitud = abs(valor)

        if potencia == 0:
            termino = f"{magnitud:g}"
        else:
            factor = "" if magnitud == 1 else f"{magnitud:g}"
            variable = "s" if potencia == 1 else f"s^{{{potencia}}}"
            termino = factor + variable

        if not partes:
            partes.append(("-" if valor < 0 else "") + termino)
        else:
            partes.append((" - " if valor < 0 else " + ") + termino)

    return "".join(partes) or "0"


with st.sidebar:
    st.header("Planta G(s)")
    num_texto = st.text_input("Numerador", "1, 1")
    den_texto = st.text_input("Denominador", "1, 1, 2")
    st.caption(
        "Coeficientes en potencias descendentes de s. "
        "Usa punto para decimales e incluye los ceros."
    )

    st.header("Simulación")
    referencia = st.number_input("Referencia", value=1.0, step=0.1)
    duracion = st.number_input(
        "Duración (s)", min_value=1.0, max_value=300.0,
        value=20.0, step=1.0
    )
    ts = st.number_input(
        "Periodo de muestreo Ts (s)",
        min_value=0.001, max_value=1.0,
        value=0.1, step=0.01, format="%.3f"
    )
    tf = st.number_input(
        "Filtro derivativo Tf (s)",
        min_value=0.001, max_value=10.0,
        value=0.1, step=0.01, format="%.3f"
    )

st.subheader("Ganancias del controlador")
modo = st.radio(
    "Forma de ajuste", ["Valores exactos", "Deslizadores"], horizontal=True
)

columnas = st.columns(3)
ganancias = []

for columna, nombre, inicial in zip(
    columnas, ["Kp", "Ki", "Kd"], [2.0, 1.0, 0.1]
):
    with columna:
        if modo == "Valores exactos":
            valor = st.number_input(
                nombre, min_value=0.0, value=inicial,
                step=0.01, format="%.2f", key=f"numero_{nombre}"
            )
        else:
            valor = st.slider(
                nombre, min_value=0.0, max_value=20.0,
                value=inicial, step=0.01, key=f"slider_{nombre}"
            )
        ganancias.append(valor)

kp, ki, kd = ganancias

st.caption(
    "Realimentación negativa unitaria y condiciones iniciales cero. "
    "La derivada se aplica sobre la salida y se filtra para evitar "
    "un pico derivativo al cambiar la referencia. Sin saturación del actuador."
)

try:
    num = leer_coeficientes(num_texto)
    den = leer_coeficientes(den_texto)

    if den.size < 2:
        raise ValueError("El denominador debe tener grado de al menos 1.")
    if num.size == 0:
        raise ValueError("El numerador no puede ser completamente cero.")
    if num.size >= den.size:
        raise ValueError(
            "Esta versión requiere una planta estrictamente propia: "
            "grado del numerador menor que el del denominador."
        )

    st.latex(r"G(s)=\frac{" + polinomio(num) + "}{" + polinomio(den) + "}")

    cantidad = int(np.floor(duracion / ts)) + 1
    if cantidad > 50000:
        raise ValueError("Reduce la duración o aumenta Ts: máximo 50 000 muestras.")

    # Planta discretizada con retención de orden cero (ZOH).
    a, b, c, d = tf2ss(num, den)
    ad, bd, cd, dd, _ = cont2discrete((a, b, c, d), ts, method="zoh")

    t = np.arange(cantidad) * ts
    r = np.full(cantidad, referencia)
    y = np.zeros(cantidad)
    u = np.zeros(cantidad)

    proporcional = np.zeros(cantidad)
    integral = np.zeros(cantidad)
    derivativa = np.zeros(cantidad)

    estado = np.zeros(ad.shape[0])
    acumulado = 0.0
    derivada = 0.0
    y_anterior = 0.0
    alpha = tf / (tf + ts)
    interrumpida = False

    for k in range(cantidad):
        salida = float((cd @ estado).item())
        error = r[k] - salida

        # Derivada filtrada de la medida.
        derivada = (
            alpha * derivada
            + (salida - y_anterior) / (tf + ts)
        )

        p = kp * error
        i = ki * acumulado
        deriv = -kd * derivada
        control = p + i + deriv

        if (
            not np.all(np.isfinite([salida, control]))
            or max(abs(salida), abs(control)) > 1e8
        ):
            interrumpida = True
            t, r, y, u = t[:k], r[:k], y[:k], u[:k]
            proporcional = proporcional[:k]
            integral = integral[:k]
            derivativa = derivativa[:k]
            break

        y[k] = salida
        u[k] = control
        proporcional[k] = p
        integral[k] = i
        derivativa[k] = deriv

        # u[k] actúa durante el siguiente intervalo de muestreo.
        estado = ad @ estado + bd[:, 0] * control
        acumulado += error * ts
        y_anterior = salida

    if interrumpida:
        st.warning(
            "Simulación detenida por valores excesivos. "
            "Revisa las ganancias, la planta y el periodo de muestreo. "
            "Las gráficas muestran únicamente el tramo calculado."
        )

    if len(t) == 0:
        raise ValueError("No se pudo calcular una respuesta con estos valores.")

    def grafica(titulo, series, escalonada=False):
        fig = go.Figure()
        for nombre, datos, color in series:
            fig.add_trace(go.Scatter(
                x=t,
                y=datos,
                name=nombre,
                mode="lines",
                line=dict(
                    color=color,
                    width=2.5,
                    shape="hv" if escalonada else "linear"
                )
            ))
        fig.update_layout(
            title=titulo,
            xaxis_title="Tiempo (s)",
            yaxis_title="Amplitud",
            height=420,
            margin=dict(l=30, r=20, t=60, b=40),
            legend=dict(orientation="h", y=1.12),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    grafica("Referencia y salida", [
        ("Referencia r(t)", r, "#e69f00"),
        ("Salida y(t)", y, "#0072b2")
    ])

    grafica("Entrada a la planta: acción de control", [
        ("Control u(t)", u, "#009e73")
    ], escalonada=True)

    with st.expander("Ver las contribuciones proporcional, integral y derivativa"):
        grafica("Componentes del PID", [
            ("Proporcional", proporcional, "#0072b2"),
            ("Integral", integral, "#009e73"),
            ("Derivativa", derivativa, "#cc79a7")
        ], escalonada=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Última salida calculada", f"{y[-1]:.4f}")
    m2.metric("Último error calculado", f"{r[-1] - y[-1]:.4f}")
    m3.metric("Máximo |u| calculado", f"{np.max(np.abs(u)):.4f}")

    st.caption(
        "Estos valores corresponden al intervalo simulado; "
        "no garantizan que se haya alcanzado el estado estacionario. "
        "Cada cambio recalcula la simulación desde cero."
    )

except (ValueError, OverflowError, np.linalg.LinAlgError) as error:
    st.error(str(error))
streamlit
numpy
scipy
plotly
