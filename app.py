import io
import csv
import numpy as np
import streamlit as st
import plotly.graph_objects as go
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
        opacity: 0.10;
        background:
            repeating-linear-gradient(
                0deg,
                rgba(255,255,255,0.15) 0px,
                rgba(255,255,255,0.15) 1px,
                transparent 1px,
                transparent 4px
            );
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
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
        color: white !important;
        text-shadow:
            0 0 5px #ff2bd6,
            0 0 14px #ff2bd6,
            0 0 28px #742cff;
    }

    h2, h3 {
        color: #5ffcff !important;
        text-shadow: 0 0 8px #00d9ff;
    }

    p {
        color: #e7c8ff !important;
    }

    .retro-header {
        padding: 28px;
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
        font-size: 2.5rem;
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
        box-shadow: 0 0 10px rgba(95,252,255,0.55);
    }

    div[data-testid="stMetricValue"] {
        color: #5ffcff !important;
        text-shadow: 0 0 8px #00d9ff;
        font-family: 'Orbitron', sans-serif;
    }

    .stTextInput input,
    .stNumberInput input {
        background-color: #0e1027 !important;
        color: white !important;
        border: 1px solid #a633ff !important;
        border-radius: 10px !important;
    }

    div[data-testid="stExpander"] {
        background: rgba(14, 16, 39, 0.85);
        border: 1px solid #ff2bd6;
        border-radius: 15px;
    }

    .stButton > button {
        background: linear-gradient(90deg, #ff2bd6, #742cff);
        color: white;
        border: 1px solid #5ffcff;
        border-radius: 10px;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        box-shadow: 0 0 12px rgba(255,43,214,0.7);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 10px #5ffcff, 0 0 22px #ff2bd6;
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
        raise ValueError("Ingresa coeficientes válidos.")

    if not np.all(np.isfinite(valores)):
        raise ValueError("Los coeficientes deben ser finitos.")

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


def simular(
    num,
    den,
    referencia,
    duracion,
    ts,
    tf,
    kp,
    ki,
    kd,
    beta,
    gamma=0.0
):
    cantidad = int(
        np.floor(duracion / ts)
    ) + 1

    if cantidad > 50000:
        raise ValueError(
            "Reduce la duración o aumenta Ts."
        )

    a, b, c, d = tf2ss(num, den)

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

    estado = np.zeros(ad.shape[0])
    integral_acumulada = 0.0
    derivada_filtrada = 0.0
    entrada_derivativa_anterior = 0.0

    alpha = tf / (tf + ts)
    interrumpida = False

    for k in range(cantidad):

        salida = float(
            (cd @ estado).item()
        )

        error = referencia_array[k] - salida
        error_proporcional = (
            beta * referencia_array[k] - salida
        )

        entrada_derivativa = gamma * referencia_array[k] - salida
        derivada_filtrada = (
            alpha * derivada_filtrada
            + (entrada_derivativa - entrada_derivativa_anterior)
            / (tf + ts)
        )

        p = kp * error_proporcional
        i = ki * integral_acumulada
        deriv = kd * derivada_filtrada

        control = p + i + deriv

        if (
            not np.all(
                np.isfinite([salida, control, p, i, deriv])
            )
            or max(abs(salida), abs(control)) > 1e8
        ):
            interrumpida = True
            ultimo = k
            break

        salida_array[k] = salida
        control_array[k] = control
        error_array[k] = error

        proporcional_array[k] = p
        integral_array[k] = i
        derivativa_array[k] = deriv

        estado = ad @ estado + bd[:, 0] * control

        integral_acumulada += error * ts
        entrada_derivativa_anterior = entrada_derivativa

    if interrumpida:

        tiempo = tiempo[:ultimo]
        referencia_array = referencia_array[:ultimo]
        salida_array = salida_array[:ultimo]
        control_array = control_array[:ultimo]
        error_array = error_array[:ultimo]

        proporcional_array = proporcional_array[:ultimo]
        integral_array = integral_array[:ultimo]
        derivativa_array = derivativa_array[:ultimo]

    if len(tiempo) == 0:
        raise ValueError(
            "No se pudo calcular la respuesta."
        )

    error_final = (
        referencia_array[-1] - salida_array[-1]
    )

    sobreimpulso = (
        max(0.0, (np.max(salida_array / referencia) - 1.0) * 100.0)
        if referencia != 0 else None
    )

    banda = max(
        0.02 * abs(referencia),
        0.01
    )

    fuera = np.where(
        np.abs(
            referencia_array - salida_array
        ) > banda
    )[0]

    if len(fuera) == 0:

        tiempo_establecimiento = 0.0

    elif fuera[-1] < len(tiempo) - 1:

        tiempo_establecimiento = (
            tiempo[fuera[-1] + 1]
        )

    else:
        tiempo_establecimiento = None

    if interrumpida:
        tiempo_establecimiento = None

    return {
        "t": tiempo,
        "r": referencia_array,
        "y": salida_array,
        "u": control_array,
        "e": error_array,
        "p": proporcional_array,
        "i": integral_array,
        "d": derivativa_array,
        "error_final": error_final,
        "sobreimpulso": sobreimpulso,
        "ts_establecimiento": tiempo_establecimiento,
        "interrumpida": interrumpida
    }


def estilo(figura, titulo):

    figura.update_layout(
        title=dict(
            text=titulo,
            font=dict(
                family="Orbitron",
                size=20,
                color="#5ffcff"
            )
        ),
        height=460,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,8,25,0.92)",
        font=dict(color="#f1d9ff"),
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

tipo_pid = st.selectbox(
    "Tipo de controlador",
    ["PID clásico (1DOF)", "PID con derivada sobre la salida (PI-D)",
     "PID 2DOF — ponderación de referencia"],
    index=1
)
ponderado = tipo_pid.startswith("PID 2DOF")
if ponderado:
    col1, col2 = st.columns(2)
    with col1:
        beta = st.number_input("β — ponderación proporcional", 0.0, 1.0, 1.0, 0.01)
    with col2:
        gamma = st.number_input("γ — ponderación derivativa", 0.0, 1.0, 0.0, 0.01)
    st.caption("β pondera la referencia en P; γ la pondera en D. La integral utiliza r − y.")
else:
    beta = 1.0
    gamma = 1.0 if tipo_pid.startswith("PID clásico") else 0.0
    st.caption(f"Ponderaciones de este modo: β = {beta:g}, γ = {gamma:g}.")

st.latex(r"U(s)=K_p[\beta R(s)-Y(s)]+\frac{K_i}{s}[R(s)-Y(s)]"
         r"+\frac{K_d s}{1+T_f s}[\gamma R(s)-Y(s)]")
st.caption("Referencia escalón desde cero, realimentación unitaria y condiciones iniciales cero. "
           "Planta con retención ZOH; integral rectangular y derivada filtrada por Euler hacia atrás. "
           "Sin saturación del actuador. Cada ajuste recalcula la simulación.")

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
# BOTONES
# =====================================================

col_a, col_b = st.columns(2)

with col_a:

    iniciar = st.button(
        "▶ INICIAR SIMULACIÓN",
        use_container_width=True
    )

with col_b:

    limpiar = st.button(
        "⏹ LIMPIAR",
        use_container_width=True
    )


if "ejecutar" not in st.session_state:
    st.session_state.ejecutar = False


if iniciar:
    st.session_state.ejecutar = True


if limpiar:

    st.session_state.ejecutar = False
    st.rerun()


if not st.session_state.ejecutar:

    st.info(
        "Configura la planta y pulsa "
        "INICIAR SIMULACIÓN."
    )

    st.stop()


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
        raise ValueError("El numerador no puede ser completamente cero.")

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

    resultado = simular(
        num,
        den,
        referencia,
        duracion,
        ts,
        tf,
        kp,
        ki,
        kd,
        beta,
        gamma
    )

    tiempo = resultado["t"]
    r = resultado["r"]
    y = resultado["y"]
    u = resultado["u"]
    e = resultado["e"]
    p = resultado["p"]
    i = resultado["i"]
    d = resultado["d"]


    if resultado["interrumpida"]:
        st.warning("Simulación interrumpida por valores excesivos o no finitos. "
                   "Se muestra solo el tramo calculado. Revisa ganancias, planta y Ts.")
    else:
        st.success("Simulación completada")
    st.caption("Completar el intervalo simulado no demuestra estabilidad del sistema.")

    # Cuatro paneles independientes. Las ecuaciones corresponden al cálculo discreto.
    def mostrar_grafica(titulo, series, clave, escalonada=False):
        figura = go.Figure()
        for nombre, datos, color, ecuacion in series:
            figura.add_trace(go.Scatter(
                x=tiempo, y=datos, name=nombre, mode="lines",
                line=dict(color=color, width=3,
                          shape="hv" if escalonada else "linear"),
                hovertemplate=(
                    "<b>" + nombre + "</b><br>"
                    "Tiempo: %{x:.3f} s<br>"
                    "Valor: %{y:.6f}<br>"
                    "Ecuación: " + ecuacion + "<extra></extra>"
                )
            ))
        estilo(figura, titulo)
        figura.update_layout(uirevision=clave)
        figura.update_yaxes(title_text="Amplitud")
        figura.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
        st.plotly_chart(figura, use_container_width=True, key=clave,
                        config={"scrollZoom": True, "displaylogo": False})

    st.caption("Arrastra para ampliar; doble clic para restablecer los ejes. "
               "Pulsa una leyenda para ocultar o mostrar una señal.")
    mostrar_grafica("01 · ENTRADA DE REFERENCIA Y SALIDA", [
        ("Referencia r(t)", r, "#ffb000", f"r[k] = {referencia:g}"),
        ("Salida y(t)", y, "#5ffcff", "y[k] = C_d x[k]")
    ], "entrada_salida")
    st.caption("Aquí la entrada es la referencia del lazo. La entrada física de la planta "
               "es u(t), representada en la siguiente gráfica.")

    mostrar_grafica("02 · SEÑAL DE CONTROL", [
        ("Control u(t)", u, "#ff2bd6", "u[k] = P[k] + I[k] + D[k]")
    ], "control", escalonada=True)

    mostrar_grafica("03 · ERROR DE CONTROL", [
        ("Error e(t)", e, "#ff625f", "e[k] = r[k] − y[k]")
    ], "error")

    mostrar_grafica("04 · COMPONENTES DEL CONTROLADOR", [
        (f"Proporcional · Kp={kp:g}", p, "#5ffcff",
         f"P[k] = {kp:g} · ({beta:g} r[k] − y[k])"),
        (f"Integral · Ki={ki:g}", i, "#28d7a2",
         f"I[k] = I[k−1] + {ki:g} · {ts:g} · e[k−1]"),
        (f"Derivativa · Kd={kd:g}", d, "#e879c9",
         "D[k] = α D[k−1] + Kd (q[k] − q[k−1]) / (Tf + Ts)"
         f"<br>q[k] = {gamma:g} r[k] − y[k]; α = Tf / (Tf + Ts)")
    ], "componentes", escalonada=True)
    st.caption("Kp, Ki y Kd son ganancias. Las curvas muestran las acciones P, I y D "
               "que generan esas ganancias a lo largo del tiempo.")

    # =================================================
    # INDICADORES
    # =================================================

    st.subheader("📊 INDICADORES DEL SISTEMA")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Salida final",
        f"{y[-1]:.4f}"
    )

    m2.metric(
        "Último error calculado",
        f"{resultado['error_final']:.4f}"
    )

    m3.metric(
        "Sobreimpulso",
        f"{resultado['sobreimpulso']:.2f}%" if resultado["sobreimpulso"] is not None else "No aplica"
    )

    m4.metric(
        "Tiempo de establecimiento",
        f"{resultado['ts_establecimiento']:.2f} s" if resultado["ts_establecimiento"] is not None else "No alcanzado"
    )


    # =================================================
    # EXPORTACIÓN CSV PARA EXCEL
    # =================================================

    st.caption("Indicadores del intervalo calculado. Establecimiento observado: banda "
               "±máx(2 % de |referencia|, 0.01). No garantiza permanencia futura en la banda.")

    st.subheader("💾 EXPORTAR RESULTADOS")

    decimal_csv = st.selectbox("Separador decimal del CSV", ["Coma (Excel en español)", "Punto"])
    archivo = io.StringIO()

    escritor = csv.writer(
        archivo,
        delimiter=";",
        lineterminator="\n"
    )

    escritor.writerow([
        "Tiempo (s)",
        "Referencia",
        "Salida",
        "Control",
        "Error",
        "Proporcional",
        "Integral",
        "Derivativa"
    ])

    for fila in zip(
        tiempo,
        r,
        y,
        u,
        e,
        p,
        i,
        d
    ):
        escritor.writerow([
            f"{valor:.6f}".replace(".", ",") if decimal_csv.startswith("Coma")
            else f"{valor:.6f}" for valor in fila
        ])

    st.download_button(
        label="⬇️ DESCARGAR RESULTADOS PARA EXCEL",
        data=archivo.getvalue().encode("utf-8-sig"),
        file_name="resultados_pid.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.caption(
        "El archivo utiliza punto y coma como separador "
        "y seis decimales. Si Excel no separa las columnas, importa desde Datos → Desde texto/CSV "
        "y selecciona punto y coma como delimitador."
    )


except (
    ValueError,
    OverflowError,
    np.linalg.LinAlgError
) as error:

    st.error(str(error))
