import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import tf2ss, cont2discrete


st.set_page_config(
    page_title="Calculadora PID",
    page_icon="⚙️",
    layout="wide"
)


# =========================
# ESTÉTICA DE LA APLICACIÓN
# =========================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                #182848 0%,
                #0b1020 45%,
                #070a13 100%
            );
        color: #f5f7ff;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #111936 0%,
            #0b1020 100%
        );
        border-right: 1px solid #26345f;
    }

    h1 {
        font-size: 3rem !important;
        font-weight: 800 !important;
        color: #ffffff;
        letter-spacing: -1px;
    }

    h2, h3 {
        color: #dce7ff;
    }

    p, label {
        color: #b9c6e4 !important;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(
            135deg,
            rgba(35, 52, 100, 0.95),
            rgba(20, 29, 62, 0.95)
        );
        border: 1px solid #3b55a0;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.25);
    }

    div[data-testid="stMetricLabel"] {
        color: #9eb4e8 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #65d9ff !important;
        font-weight: 800;
    }

    .stButton > button {
        background: linear-gradient(
            90deg,
            #5b5ff0,
            #9b51e0
        );
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        padding: 0.6rem 1rem;
        transition: 0.25s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(115, 98, 255, 0.45);
    }

    .stTextInput input,
    .stNumberInput input {
        background: #11182d !important;
        color: #ffffff !important;
        border: 1px solid #3d518c !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #11182d;
        border: 1px solid #3d518c;
        border-radius: 10px;
    }

    div[data-testid="stExpander"] {
        background: rgba(20, 30, 65, 0.75);
        border: 1px solid #344b88;
        border-radius: 16px;
    }

    hr {
        border-color: #293a70;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# FUNCIONES
# =========================

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

    if valores.size == 0 or not np.all(np.isfinite(valores)):
        raise ValueError(
            "Ingresa coeficientes numéricos válidos."
        )

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


def grafica(titulo, tiempo, series, escalonada=False):
    figura = go.Figure()

    for nombre, datos, color in series:
        figura.add_trace(
            go.Scatter(
                x=tiempo,
                y=datos,
                name=nombre,
                mode="lines",
                line=dict(
                    color=color,
                    width=3,
                    shape="hv" if escalonada else "linear"
                )
            )
        )

    figura.update_layout(
        title=dict(
            text=titulo,
            font=dict(
                size=22,
                color="#dce7ff"
            )
        ),
        xaxis_title="Tiempo (s)",
        yaxis_title="Amplitud",
        height=450,
        margin=dict(
            l=30,
            r=20,
            t=70,
            b=40
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(9,15,32,0.75)",
        font=dict(color="#dce7ff"),
        legend=dict(
            orientation="h",
            y=1.12
        ),
        hovermode="x unified",
        xaxis=dict(
            gridcolor="#28365e",
            zerolinecolor="#526ca8"
        ),
        yaxis=dict(
            gridcolor="#28365e",
            zerolinecolor="#526ca8"
        )
    )

    st.plotly_chart(
        figura,
        use_container_width=True
    )


# =========================
# ENCABEZADO
# =========================

st.markdown(
    """
    <div style="
        padding: 28px;
        border-radius: 22px;
        margin-bottom: 25px;
        background: linear-gradient(
            135deg,
            rgba(40, 57, 120, 0.95),
            rgba(92, 48, 145, 0.90)
        );
        box-shadow: 0 12px 35px rgba(0,0,0,0.3);
    ">
        <h1 style="margin-bottom: 8px;">
            ⚙️ Calculadora PID
        </h1>

        <p style="
            font-size: 1.1rem;
            color: #dbe5ff !important;
            margin-bottom: 0;
        ">
            Simula plantas, ajusta Kp, Ki y Kd,
            y analiza la respuesta del sistema.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# PANEL LATERAL
# =========================

with st.sidebar:

    st.header("🧩 Planta G(s)")

    num_texto = st.text_input(
        "Numerador",
        "1, 1"
    )

    den_texto = st.text_input(
        "Denominador",
        "1, 1, 2"
    )

    st.caption(
        "Escribe los coeficientes en potencias descendentes "
        "de s. Usa punto para decimales."
    )

    st.divider()

    st.header("⏱️ Simulación")

    referencia = st.number_input(
        "Referencia",
        value=1.0,
        step=0.1
    )

    duracion = st.number_input(
        "Duración de simulación (s)",
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


# =========================
# TIPO DE CONTROLADOR
# =========================

st.subheader("🎛️ Tipo de controlador")

tipo_pid = st.radio(
    "Selecciona el tipo de PID",
    [
        "PID clásico (1 grado de libertad)",
        "PID de dos grados de libertad"
    ],
    horizontal=True
)


col_beta, col_estado = st.columns(2)

with col_beta:

    beta = st.number_input(
        "Ponderación de la referencia β",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
        format="%.2f",
        help=(
            "β=1 equivale al PID clásico. "
            "Valores menores reducen la acción proporcional "
            "ante cambios de referencia."
        )
    )


with col_estado:

    if tipo_pid == "PID clásico (1 grado de libertad)":

        beta = 1.0

        st.success(
            "PID clásico activo. β = 1"
        )

    else:

        st.info(
            "PID de dos grados de libertad activo."
        )


# =========================
# GANANCIAS
# =========================

st.subheader("📈 Ganancias del controlador")

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


st.caption(
    "Realimentación negativa unitaria. "
    "El PID 2DOF usa β para ponderar la referencia "
    "en la acción proporcional. "
    "La derivada se aplica sobre la salida para evitar "
    "un pico derivativo ante un cambio de referencia."
)


# =========================
# SIMULACIÓN
# =========================

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
            "La planta debe ser estrictamente propia: "
            "el grado del numerador debe ser menor que "
            "el grado del denominador."
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
            "Reduce la duración o aumenta Ts. "
            "Máximo 50 000 muestras."
        )

    # Conversión de la planta a espacio de estados
    a, b, c, d = tf2ss(num, den)

    # Discretización mediante ZOH
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

        # Salida actual de la planta
        salida = float(
            (cd @ estado_planta).item()
        )

        # Error completo para la acción integral
        error_integral = (
            referencia_array[k] - salida
        )

        # Error proporcional
        if tipo_pid == "PID de dos grados de libertad":

            error_proporcional = (
                beta * referencia_array[k] - salida
            )

        else:

            error_proporcional = error_integral

        # Derivada filtrada sobre la salida
        derivada_filtrada = (
            alpha * derivada_filtrada
            + (salida - salida_anterior)
            / (tf + ts)
        )

        # Componentes del PID
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

        # Protección numérica
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

            proporcional_array = proporcional_array[:k]
            integral_array = integral_array[:k]
            derivativa_array = derivativa_array[:k]

            break

        salida_array[k] = salida
        control_array[k] = control

        proporcional_array[k] = componente_p
        integral_array[k] = componente_i
        derivativa_array[k] = componente_d

        # Actualización de la planta
        estado_planta = (
            ad @ estado_planta
            + bd[:, 0] * control
        )

        # La integral utiliza el error completo
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

    # =========================
    # GRÁFICAS
    # =========================

    grafica(
        "Respuesta de la planta",
        tiempo,
        [
            (
                "Referencia r(t)",
                referencia_array,
                "#ffb000"
            ),
            (
                "Salida y(t)",
                salida_array,
                "#48c9ff"
            )
        ]
    )

    grafica(
        "Acción de control",
        tiempo,
        [
            (
                "Control u(t)",
                control_array,
                "#28d7a2"
            )
        ],
        escalonada=True
    )

    with st.expander(
        "🔍 Ver componentes proporcional, integral y derivativa"
    ):

        grafica(
            "Componentes del controlador",
            tiempo,
            [
                (
                    "Proporcional",
                    proporcional_array,
                    "#48c9ff"
                ),
                (
                    "Integral",
                    integral_array,
                    "#28d7a2"
                ),
                (
                    "Derivativa",
                    derivativa_array,
                    "#e879c9"
                )
            ],
            escalonada=True
        )

    # =========================
    # INDICADORES
    # =========================

    st.subheader("📊 Indicadores de desempeño")

    metricas = st.columns(3)

    metricas[0].metric(
        "Última salida calculada",
        f"{salida_array[-1]:.4f}"
    )

    metricas[1].metric(
        "Último error calculado",
        f"{referencia_array[-1] - salida_array[-1]:.4f}"
    )

    metricas[2].metric(
        "Máximo |u| calculado",
        f"{np.max(np.abs(control_array)):.4f}"
    )

    st.caption(
        "Cada cambio en la planta, referencia, ganancias "
        "o tipo de controlador recalcula la simulación "
        "desde cero."
    )


except (
    ValueError,
    OverflowError,
    np.linalg.LinAlgError
) as error:

    st.error(str(error))
