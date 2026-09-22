import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import tf2ss, cont2discrete


st.set_page_config(
    page_title="PID TV Control",
    page_icon="📺",
    layout="wide"
)


# =====================================================
# ESTÉTICA RETROFUTURISTA
# =====================================================

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&display=swap'
    );

    .stApp {
        background:
            radial-gradient(
                circle at 50% -20%,
                #3b0b55 0%,
                #180c32 35%,
                #090b1b 72%,
                #03040b 100%
            );
        color: #f7e9ff;
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 999;
        opacity: 0.12;
        background:
            repeating-linear-gradient(
                0deg,
                rgba(255,255,255,0.12) 0px,
                rgba(255,255,255,0.12) 1px,
                transparent 1px,
                transparent 4px
            );
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #160d2e 0%,
                #08091b 100%
            );
        border-right: 2px solid #ff2bd6;
    }

    h1, h2, h3, label {
        font-family: 'Orbitron', sans-serif !important;
    }

    h1 {
        color: #ffffff !important;
        text-shadow:
            0 0 5px #ff2bd6,
            0 0 14px #ff2bd6,
            0 0 28px #742cff;
        letter-spacing: 2px;
    }

    h2, h3 {
        color: #5ffcff !important;
        text-shadow: 0 0 8px #00d9ff;
    }

    p {
        color: #e7c8ff !important;
    }

    .retro-header {
        padding: 30px;
        margin-bottom: 25px;
        border: 2px solid #ff2bd6;
        border-radius: 20px;
        background:
            linear-gradient(
                135deg,
                rgba(49, 14, 81, 0.98),
                rgba(11, 24, 67, 0.98)
            );
        box-shadow:
            0 0 10px #ff2bd6,
            0 0 30px rgba(255,43,214,0.45),
            inset 0 0 25px rgba(95,252,255,0.12);
    }

    .retro-header h1 {
        margin: 0 0 8px 0;
        font-size: 2.6rem;
    }

    .retro-header p {
        margin: 0;
        font-size: 1rem;
        letter-spacing: 1px;
    }

    div[data-testid="stMetric"] {
        background:
            linear-gradient(
                135deg,
                rgba(22, 17, 62, 0.95),
                rgba(34, 9, 55, 0.95)
            );
        border: 1px solid #5ffcff;
        border-radius: 15px;
        padding: 18px;
        box-shadow:
            0 0 9px rgba(95,252,255,0.7),
            inset 0 0 15px rgba(255,43,214,0.12);
    }

    div[data-testid="stMetricLabel"] {
        color: #d8b7ff !important;
    }

    div[data-testid="stMetricValue"] {
        color: #5ffcff !important;
        text-shadow: 0 0 8px #00d9ff;
        font-family: 'Orbitron', sans-serif;
    }

    .stTextInput input,
    .stNumberInput input {
        background-color: #0e1027 !important;
        color: #ffffff !important;
        border: 1px solid #a633ff !important;
        border-radius: 10px !important;
        box-shadow: inset 0 0 8px rgba(166,51,255,0.3);
    }

    div[data-baseweb="select"] > div {
        background-color: #0e1027;
        border: 1px solid #a633ff;
        border-radius: 10px;
    }

    .stButton > button {
        background:
            linear-gradient(
                90deg,
                #ff2bd6,
                #742cff
            );
        color: white;
        border: 1px solid #5ffcff;
        border-radius: 10px;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        box-shadow: 0 0 12px rgba(255,43,214,0.7);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow:
            0 0 10px #5ffcff,
            0 0 22px #ff2bd6;
    }

    div[data-testid="stExpander"] {
        background: rgba(14, 16, 39, 0.85);
        border: 1px solid #ff2bd6;
        border-radius: 15px;
        box-shadow: 0 0 12px rgba(255,43,214,0.35);
    }

    hr {
        border-color: #a633ff;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# FUNCIONES
# =====================================================

def leer_coeficientes(texto):
    texto = (
        texto
        .replace("[", "")
        .replace("]", "")
        .replace(",", " ")
    )

    valores = np.array(
        [float(x) for x in texto.split()]
    )

    if valores.size == 0:
        raise ValueError(
            "Ingresa coeficientes numéricos válidos."
        )

    if not np.all(np.isfinite(valores)):
        raise ValueError(
            "Los coeficientes deben ser finitos."
        )

    return np.trim_zeros(valores, "f")


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
            variable = (
                "s"
                if potencia == 1
                else f"s^{{{potencia}}}"
            )
            termino = factor + variable

        if not partes:
            partes.append(
                ("-" if valor < 0 else "") + termino
            )
        else:
            partes.append(
                (" - " if valor < 0 else " + ") + termino
            )

    return "".join(partes) or "0"


def estilo_grafica(figura, titulo):
    figura.update_layout(
        title=dict(
            text=titulo,
            font=dict(
                family="Orbitron",
                size=20,
                color="#5ffcff"
            )
        ),
        height=470,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,8,25,0.90)",
        font=dict(
            color="#f1d9ff",
            family="Arial"
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.12
        ),
        margin=dict(
            l=45,
            r=35,
            t=80,
            b=50
        )
    )

    figura.update_xaxes(
        title_text="Tiempo (s)",
        gridcolor="#34245c",
        zerolinecolor="#ff2bd6"
    )

    figura.update_yaxes(
        gridcolor="#34245c",
        zerolinecolor="#5ffcff"
    )

    return figura


# =====================================================
# ENCABEZADO
# =====================================================

st.markdown(
    """
    <div class="retro-header">
        <h1>📺 PID TV CONTROL</h1>
        <p>
            LABORATORIO RETROFUTURISTA DE CONTROL AUTOMÁTICO
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("🧩 PLANTA G(s)")

    num_texto = st.text_input(
        "Numerador",
        "1, 1"
    )

    den_texto = st.text_input(
        "Denominador",
        "1, 1, 2"
    )

    st.caption(
        "Coeficientes en potencias descendentes de s."
    )

    st.divider()

    st.header("⏱️ SIMULACIÓN")

    referencia = st.number_input(
        "Referencia",
        value=1.0,
        step=0.1
    )

    duracion = st.number_input(
        "Tiempo de simulación (s)",
        min_value=1.0,
        max_value=300.0,
        value=20.0,
        step=1.0
    )

    ts = st.number_input(
        "Periodo de muestreo Ts (s)",
        min_value=0.001,
        max_value=1.0,
        value=0.1,
        step=0.01,
        format="%.3f"
    )

    tf = st.number_input(
        "Filtro derivativo Tf (s)",
        min_value=0.001,
        max_value=10.0,
        value=0.1,
        step=0.01,
        format="%.3f"
    )


# =====================================================
# CONTROLADOR
# =====================================================

st.subheader("🎛️ CONFIGURACIÓN DEL CONTROLADOR")

tipo_pid = st.radio(
    "Tipo de controlador",
    [
        "PID clásico (1 grado de libertad)",
        "PID de dos grados de libertad"
    ],
    horizontal=True
)

col_beta, col_info = st.columns(2)

with col_beta:

    beta = st.number_input(
        "Ponderación de referencia β",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
        format="%.2f",
        help=(
            "β=1 equivale al PID clásico. "
            "Valores menores reducen la respuesta "
            "proporcional ante cambios de referencia."
        )
    )


with col_info:

    if tipo_pid == "PID clásico (1 grado de libertad)":
        beta = 1.0
        st.success("PID CLÁSICO ACTIVO | β = 1")

    else:
        st.info("PID 2DOF ACTIVO | β CONFIGURABLE")


st.subheader("📈 GANANCIAS")

modo = st.radio(
    "Forma de ajuste",
    [
        "Valores exactos",
        "Deslizadores"
    ],
    horizontal=True
)

columnas = st.columns(3)
ganancias = []

for columna, nombre, inicial in zip(
    columnas,
    ["Kp", "Ki", "Kd"],
    [2.0, 1.0, 0.1]
):

    with columna:

        if modo == "Valores exactos":

            valor = st.number_input(
                nombre,
                min_value=0.0,
                value=inicial,
                step=0.01,
                format="%.2f",
                key=f"numero_{nombre}"
            )

        else:

            valor = st.slider(
                nombre,
                min_value=0.0,
                max_value=20.0,
                value=inicial,
                step=0.01,
                key=f"slider_{nombre}"
            )

        ganancias.append(valor)


kp, ki, kd = ganancias


# =====================================================
# SIMULACIÓN
# =====================================================

try:

    num = leer_coeficientes(num_texto)
    den = leer_coeficientes(den_texto)

    if den.size < 2:
        raise ValueError(
            "El denominador debe tener grado de al menos 1."
        )

    if num.size == 0:
        raise ValueError(
            "El numerador no puede ser completamente cero."
        )

    if num.size >= den.size:
        raise ValueError(
            "La planta debe ser estrictamente propia."
        )

    st.latex(
        r"G(s)=\frac{"
        + polinomio(num)
        + "}{"
        + polinomio(den)
        + "}"
    )

    cantidad = int(
        np.floor(duracion / ts)
    ) + 1

    if cantidad > 50000:
        raise ValueError(
            "Reduce la duración o aumenta Ts."
        )

    # Planta en espacio de estados
    a, b, c, d = tf2ss(num, den)

    # Discretización ZOH
    ad, bd, cd, dd, _ = cont2discrete(
        (a, b, c, d),
        ts,
        method="zoh"
    )

    tiempo = np.arange(cantidad) * ts

    referencia_array = np.full(
        cantidad,
        referencia
    )

    salida_array = np.zeros(cantidad)
    control_array = np.zeros(cantidad)
    error_array = np.zeros(cantidad)

    proporcional_array = np.zeros(cantidad)
    integral_array = np.zeros(cantidad)
    derivativa_array = np.zeros(cantidad)

    estado_planta = np.zeros(
        ad.shape[0]
    )

    acumulado_integral = 0.0
    derivada_filtrada = 0.0
    salida_anterior = 0.0

    alpha = tf / (tf + ts)
    interrumpida = False

    for k in range(cantidad):

        salida = float(
            (cd @ estado_planta).item()
        )

        error_integral = (
            referencia_array[k] - salida
        )

        if tipo_pid == "PID de dos grados de libertad":

            error_proporcional = (
                beta * referencia_array[k] - salida
            )

        else:

            error_proporcional = error_integral

        derivada_filtrada = (
            alpha * derivada_filtrada
            + (salida - salida_anterior)
            / (tf + ts)
        )

        componente_p = (
            kp * error_proporcional
        )

        componente_i = (
            ki * acumulado_integral
        )

        componente_d = (
            -kd * derivada_filtrada
        )

        control = (
            componente_p
            + componente_i
            + componente_d
        )

        if (
            not np.all(
                np.isfinite(
                    [salida, control]
                )
            )
            or max(
                abs(salida),
                abs(control)
            ) > 1e8
        ):

            interrumpida = True

            tiempo = tiempo[:k]
            referencia_array = referencia_array[:k]
            salida_array = salida_array[:k]
            control_array = control_array[:k]
            error_array = error_array[:k]

            proporcional_array = proporcional_array[:k]
            integral_array = integral_array[:k]
            derivativa_array = derivativa_array[:k]

            break

        salida_array[k] = salida
        control_array[k] = control
        error_array[k] = error_integral

        proporcional_array[k] = componente_p
        integral_array[k] = componente_i
        derivativa_array[k] = componente_d

        estado_planta = (
            ad @ estado_planta
            + bd[:, 0] * control
        )

        acumulado_integral += (
            error_integral * ts
        )

        salida_anterior = salida

    if interrumpida:

        st.warning(
            "La simulación se detuvo por valores excesivos. "
            "Revisa las ganancias, la planta o Ts."
        )

    if len(tiempo) == 0:
        raise ValueError(
            "No se pudo calcular una respuesta válida."
        )

    # =================================================
    # GRÁFICA 1: PROCESO Y CONTROL
    # =================================================

    grafica_proceso = make_subplots(
        specs=[[{"secondary_y": True}]]
    )

    grafica_proceso.add_trace(
        go.Scatter(
            x=tiempo,
            y=referencia_array,
            name="Referencia r(t)",
            mode="lines",
            line=dict(
                color="#ffb000",
                width=3
            ),
            hovertemplate=(
                "<b>Referencia r(t)</b><br>"
                "Tiempo: %{x:.3f} s<br>"
                "Posición: %{y:.4f}<br>"
                "Ecuación: r(t) = referencia"
                "<extra></extra>"
            )
        ),
        secondary_y=False
    )

    grafica_proceso.add_trace(
        go.Scatter(
            x=tiempo,
            y=salida_array,
            name="Salida y(t)",
            mode="lines",
            line=dict(
                color="#5ffcff",
                width=3
            ),
            hovertemplate=(
                "<b>Salida y(t)</b><br>"
                "Tiempo: %{x:.3f} s<br>"
                "Posición: %{y:.4f}<br>"
                "Ecuación: y(t) = G(s) · u(t)"
                "<extra></extra>"
            )
        ),
        secondary_y=False
    )

    grafica_proceso.add_trace(
        go.Scatter(
            x=tiempo,
            y=control_array,
            name="Control u(t)",
            mode="lines",
            line=dict(
                color="#ff2bd6",
                width=3
            ),
            hovertemplate=(
                "<b>Control u(t)</b><br>"
                "Tiempo: %{x:.3f} s<br>"
                "Posición: %{y:.4f}<br>"
                "Ecuación: u(t) = P(t) + I(t) + D(t)"
                "<extra></extra>"
            )
        ),
        secondary_y=True
    )

    grafica_proceso = estilo_grafica(
        grafica_proceso,
        "SEÑALES DE PROCESO Y CONTROL"
    )

    grafica_proceso.update_yaxes(
        title_text="Referencia y salida",
        secondary_y=False
    )

    grafica_proceso.update_yaxes(
        title_text="Señal de control u(t)",
        secondary_y=True
    )

    st.plotly_chart(
        grafica_proceso,
        use_container_width=True
    )

    # =================================================
    # GRÁFICA 2: ERROR, INTEGRAL Y DERIVATIVA
    # =================================================

    grafica_acciones = go.Figure()

    grafica_acciones.add_trace(
        go.Scatter(
            x=tiempo,
            y=error_array,
            name="Error e(t)",
            mode="lines",
            line=dict(
                color="#ff625f",
                width=3
            ),
            hovertemplate=(
                "<b>Error e(t)</b><br>"
                "Tiempo: %{x:.3f} s<br>"
                "Posición: %{y:.4f}<br>"
                "Ecuación: e(t) = r(t) − y(t)"
                "<extra></extra>"
            )
        )
    )

    grafica_acciones.add_trace(
        go.Scatter(
            x=tiempo,
            y=integral_array,
            name="Acción integral I(t)",
            mode="lines",
            line=dict(
                color="#28d7a2",
                width=3
            ),
            hovertemplate=(
                "<b>Acción integral I(t)</b><br>"
                "Tiempo: %{x:.3f} s<br>"
                "Posición: %{y:.4f}<br>"
                "Ecuación: I(t) = Ki · ∫e(t)dt"
                "<extra></extra>"
            )
        )
    )

    grafica_acciones.add_trace(
        go.Scatter(
            x=tiempo,
            y=derivativa_array,
            name="Acción derivativa D(t)",
            mode="lines",
            line=dict(
                color="#e879c9",
                width=3
            ),
            hovertemplate=(
                "<b>Acción derivativa D(t)</b><br>"
                "Tiempo: %{x:.3f} s<br>"
                "Posición: %{y:.4f}<br>"
                "Ecuación: D(t) = −Kd · dy/dt"
                "<extra></extra>"
            )
        )
    )

    grafica_acciones = estilo_grafica(
        grafica_acciones,
        "ERROR Y ACCIONES DEL CONTROLADOR"
    )

    grafica_acciones.update_layout(
        xaxis_title="Tiempo (s)",
        yaxis_title="Amplitud"
    )

    st.plotly_chart(
        grafica_acciones,
        use_container_width=True
    )

    # =================================================
    # COMPONENTES DEL CONTROLADOR
    # =================================================

    with st.expander(
        "📺 VER COMPONENTES DEL CONTROLADOR"
    ):

        componentes = go.Figure()

        componentes.add_trace(
            go.Scatter(
                x=tiempo,
                y=proporcional_array,
                name="Proporcional P(t)",
                mode="lines",
                line=dict(
                    color="#5ffcff",
                    width=3
                ),
                hovertemplate=(
                    "<b>Proporcional P(t)</b><br>"
                    "Tiempo: %{x:.3f} s<br>"
                    "Posición: %{y:.4f}<br>"
                    "Ecuación: P(t) = Kp · eP(t)"
                    "<extra></extra>"
                )
            )
        )

        componentes.add_trace(
            go.Scatter(
                x=tiempo,
                y=integral_array,
                name="Integral I(t)",
                mode="lines",
                line=dict(
                    color="#28d7a2",
                    width=3
                ),
                hovertemplate=(
                    "<b>Integral I(t)</b><br>"
                    "Tiempo: %{x:.3f} s<br>"
                    "Posición: %{y:.4f}<br>"
                    "Ecuación: I(t) = Ki · ∫e(t)dt"
                    "<extra></extra>"
                )
            )
        )

        componentes.add_trace(
            go.Scatter(
                x=tiempo,
                y=derivativa_array,
                name="Derivativa D(t)",
                mode="lines",
                line=dict(
                    color="#e879c9",
                    width=3
                ),
                hovertemplate=(
                    "<b>Derivativa D(t)</b><br>"
                    "Tiempo: %{x:.3f} s<br>"
                    "Posición: %{y:.4f}<br>"
                    "Ecuación: D(t) = −Kd · dy/dt"
                    "<extra></extra>"
                )
            )
        )

        componentes = estilo_grafica(
            componentes,
            "COMPONENTES DEL CONTROLADOR"
        )

        st.plotly_chart(
            componentes,
            use_container_width=True
        )

    # =================================================
    # INDICADORES
    # =================================================

    st.subheader("📊 INDICADORES DEL SISTEMA")

    metricas = st.columns(3)

    metricas[0].metric(
        "Última salida",
        f"{salida_array[-1]:.4f}"
    )

    metricas[1].metric(
        "Último error",
        f"{error_array[-1]:.4f}"
    )

    metricas[2].metric(
        "Máximo |u(t)|",
        f"{np.max(np.abs(control_array)):.4f}"
    )

    st.caption(
        "Mueve el cursor sobre cualquier gráfica para observar "
        "el tiempo, la posición y la ecuación de cada señal."
    )


except (
    ValueError,
    OverflowError,
    np.linalg.LinAlgError
) as error:

    st.error(str(error))
