import numpy as np
import sympy as sp
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.integrate import odeint


def ecuacion_logistica(K:float, P0:float, r:float, t0:float, t:float, cant:float, scale:float, no_arrows):
    """
    Retorna una gráfica de la ecuacion logistica con su campo vectorial.

    Parámetros:
    -------
    - K: Capacidad de carga.
    - P0: Poblacion Inicial.
    - r: Tasa de crecimineto poblacional.
    - t0: Tiempo inicial.
    - t: Tiempo final.
    - cant: Las particiones para el eje temporal y espacial.
    - scale: Tamaño del vector del campo vectorial.
    - no_arrows: Condicion de activacion de mallado
    """

    # Rango de P y t
    P_values = np.linspace(0, K+5, cant)
    t_values = np.linspace(0, t, cant)

    # Crear una malla de puntos (P, t)
    T, P = np.meshgrid(t_values, P_values)

    # Definir la EDO
    dP_dt = r * P * (1 - P / K)

    # Solucion exacta de la Ecuación Logística
    funcion = K*P0*np.exp(r*t_values) / (P0*np.exp(r*t_values) + (K-P0)*np.exp(r*t0))

    # Campo vectorial: dP/dt (componente vertical)
    U = np.ones_like(T)  # Componente en t (horizontal)
    V = dP_dt           # Componente en P (vertical)


    # Crear el campo de vectores con Plotly
    if no_arrows == 'Desactivado':
        fig = go.Figure()
    else:
        fig = ff.create_quiver(
            T, P, U, V,
            scale=scale,
            line=dict(color='black', width=1),
            showlegend=False
        )

    # Crear la función logística
    fig.add_trace(
        go.Scatter(
            x = t_values,
            y = funcion,
            #mode = 'markers+lines',
            line=dict(color='blue'),
            name = 'Ecuación Logística'
        )
    )

    fig.add_trace(
        go.Scatter(
            x = [0, t],
            y = [K, K],
            mode = 'lines',
            line = dict(color='red', dash='dash'),
            name = 'Capacidad de carga'
        )
    )

    # Etiquetas para la gráfica
    fig.update_layout(
        title={
            'text':'Campo de vectores de dP/dt = rP(1 - P/k)',
            'x':0.5,
            'y':0.92,
            'xanchor':'center'
        },
        xaxis_title='Tiempo (t)',
        yaxis_title='Población (P)',
        width=800,
        template='ggplot2',
        margin=dict(l=10,r=10,t=90,b=0),
        paper_bgcolor='#f3eddf',
        legend=dict(orientation='h',y=1.1)
    )

    # contorno a la grafica
    fig.update_xaxes(
        mirror=True,
        showline=True,
        linecolor='green',
        gridcolor='gray',
        showgrid=False
    )
    fig.update_yaxes(
        mirror=True,
        showline=True,
        linecolor='green',
        gridcolor='gray',
        showgrid=False
    )

    return fig

## Funcion Logistica 
## 1. Averiguar alguna Ecuacion de algun modelo y desarrollarlo con Sympy
## 2. Mejorar la apariencia visual de la pagina (diferentes paginas, diferentes id)
## 3. Agregar un boton el cual me permita activar y desactivar el campo de vectores (OPCIONAL RETO)
## 4. Buscar ecuacion de dos parametros 'Depredador-Presa' (OPCIONAL RETO)


def ecuacion_logistica_sympy(K:float, P0:float, r:float, t0:float, t:float, cant:float, scale:float, no_arrows):
    """
    Retorna una gráfica de la ecuacion logistica con su campo vectorial usando sympy.

    Parámetros:
    -------
    - K: Capacidad de carga.
    - P0: Poblacion Inicial.
    - r: Tasa de crecimineto poblacional.
    - t0: Tiempo inicial.
    - t: Tiempo final.
    - cant: Las particiones para el eje temporal y espacial.
    - scale: Tamaño del vector del campo vectorial.
    - no_arrows: Condicion de activacion de mallado
    """
    # Rango de P y t
    P_values = np.linspace(0, K+5, cant)
    t_values = np.linspace(0, t, cant)

    # Simbolos
    ts = sp.Symbol('t')
    rs = sp.Symbol('r')
    Ks = sp.Symbol('K')

    # Funcion
    Pf = sp.Function('P')(ts)

    # Crear una malla de puntos (P, t)
    T, P = np.meshgrid(t_values, P_values)

    # Definir la EDO
    dP = Pf.diff(ts)
    rhs = rs*Pf*(1-(Pf/Ks))
    eq = sp.Eq(dP,rhs)
    edo_f = sp.lambdify([rs,Ks,Pf], rhs, 'numpy')

    # Resolucion de la EDO
    sol = sp.dsolve(eq)

    # Establecemos la solucion para P(0) = P0
    t0_s = sol.rhs.subs({'t':t0})      # Simbolo para t0 en funcion del input
    eq_init = sp.Eq(P0,t0_s)        # Definimos la ecuacion P0 = P(0)
    C1 =sp.Symbol('C1')             # Definimos la constante C1
    t0_sol = sp.solve(eq_init,C1)   # Resolvemos la ecuacion para hallar C1

    expr = sol.rhs.subs(C1,t0_sol[0]) # Conseguimos la expresion que soluciona la EDO reemplazando el valor de C1
    expr_simp = expr.simplify() # Simplificamos la expresion

    function = sp.lambdify([rs,Ks,ts], expr_simp, 'numpy') # Convertimos la expresion en una funcion lambda que acepte 3 arg
    function_mesh = function(r,K,t_values)
    edo_mesh = edo_f(r,K,P)

    # Campo vectorial: dP/dt (componente vertical)
    U = np.ones_like(T)  # Componente en t (horizontal)
    V = edo_mesh           # Componente en P (vertical)

    # Crear el campo de vectores con Plotly
    if no_arrows == 'Desactivado':
        fig = go.Figure()
    else:
        fig = ff.create_quiver(
            T, P, U, V,
            scale=scale,
            line=dict(color='black', width=1),
            showlegend=False
        )

    # Crear la función logística
    fig.add_trace(
        go.Scatter(
            x = t_values,
            y = function_mesh,
            #mode = 'markers+lines',
            line=dict(color='blue'),
            name = 'Ecuación Logística'
        )
    )

    fig.add_trace(
        go.Scatter(
            x = [0, t],
            y = [K, K],
            mode = 'lines',
            line = dict(color='red', dash='dash'),
            name = 'Capacidad de carga'
        )
    )

    # Etiquetas para la gráfica
    fig.update_layout(
        title={
            'text':'Campo de vectores de dP/dt = rP(1 - P/k)',
            'x':0.5,
            'y':0.92,
            'xanchor':'center'
        },
        xaxis_title='Tiempo (t)',
        yaxis_title='Población (P)',
        width=800,
        template='ggplot2',
        margin=dict(l=10,r=10,t=90,b=0),
        legend=dict(orientation='h',y=1.1),
        paper_bgcolor='#f3eddf',
    )

    # contorno a la grafica
    fig.update_xaxes(
        mirror=True,
        showline=True,
        linecolor='green',
        gridcolor='gray',
        showgrid=False
    )
    fig.update_yaxes(
        mirror=True,
        showline=True,
        linecolor='green',
        gridcolor='gray',
        showgrid=False
    )

    return fig


def ecuacion_logistica_umbral(K:float, P0:float, r:float, t0:float, t:float, cant:float, scale:float, no_arrows):
    """
    Retorna una gráfica de la ecuacion logistica con su campo vectorial.

    Parámetros:
    -------
    - K: Umbral de crecimiento.
    - P0: Poblacion Inicial.
    - r: Tasa de crecimineto poblacional.
    - t0: Tiempo inicial.
    - t: Tiempo final.
    - cant: Las particiones para el eje temporal y espacial.
    - scale: Tamaño del vector del campo vectorial.
    - no_arrows: Condicion de activacion de mallado
    """
    # Rango de P y t
    P_values = np.linspace(0, K*2, cant)
    t_values = np.linspace(0, t, cant)

    # Simbolos
    ts = sp.Symbol('t')
    rs = sp.Symbol('r')
    Ks = sp.Symbol('K')

    # Funcion
    Pf = sp.Function('P')(ts)

    # Crear una malla de puntos (P, t)
    T, P = np.meshgrid(t_values, P_values)

    # Definir la EDO
    dP = Pf.diff(ts)
    rhs = -rs*Pf*(1-(Pf/Ks))
    eq = sp.Eq(dP,rhs)
    edo_f = sp.lambdify([rs,Ks,Pf], rhs, 'numpy')

    # Resolucion de la EDO
    sol = sp.dsolve(eq)

    # Establecemos la solucion para P(0) = P0
    t0_s = sol.rhs.subs({'t':t0})      # Simbolo para t0 en funcion del input
    eq_init = sp.Eq(P0,t0_s)        # Definimos la ecuacion P0 = P(0)
    C1 =sp.Symbol('C1')             # Definimos la constante C1
    t0_sol = sp.solve(eq_init,C1)   # Resolvemos la ecuacion para hallar C1

    expr = sol.rhs.subs(C1,t0_sol[0]) # Conseguimos la expresion que soluciona la EDO reemplazando el valor de C1
    expr_simp = expr.simplify() # Simplificamos la expresion

    function = sp.lambdify([rs,Ks,ts], expr_simp, 'numpy') # Convertimos la expresion en una funcion lambda que acepte 3 arg
    function_mesh = function(r,K,t_values)
    edo_mesh = edo_f(r,K,P)


    # Como no existe una capacidad de carga, la funcion se comportara de manera
    # exponencial, para evitar problemas de errores de punto flotante usaremos
    # un tope maximo dependiente del valor del umbral (K)
    for i in range(len(function_mesh)):
        if function_mesh[i] > K*3 or function_mesh[i] < 0:
            function_mesh[i] = K*3

    # Campo vectorial: dP/dt (componente vertical)
    U = np.ones_like(T)  # Componente en t (horizontal)
    V = edo_mesh           # Componente en P (vertical)

    # Crear el campo de vectores con Plotly
    if no_arrows == 'Desactivado':
        fig = go.Figure()
    else:
        fig = ff.create_quiver(
            T, P, U, V,
            scale=scale,
            line=dict(color='black', width=1),
            showlegend=False
        )

    # Crear la función logística
    fig.add_trace(
        go.Scatter(
            x = t_values,
            y = function_mesh,
            #mode = 'markers+lines',
            line=dict(color='blue'),
            name = 'Ecuación Logística'
        )
    )

    fig.add_trace(
        go.Scatter(
            x = [0, t],
            y = [K, K],
            mode = 'lines',
            line = dict(color='red', dash='dash'),
            name = 'Umbral de Supervivencia'
        )
    )

    # Etiquetas para la gráfica
    fig.update_layout(
        title={
            'text':'Campo de vectores de dP/dt = -rP(1 - P/k)',
            'x':0.5,
            'y':0.92,
            'xanchor':'center'
        },
        xaxis_title='Tiempo (t)',
        yaxis_title='Población (P)',
        width=800,
        template='ggplot2',
        paper_bgcolor='#f3eddf',
        margin=dict(l=10,r=10,t=90,b=0),
        legend=dict(orientation='h',y=1.1)
    )

    # contorno a la grafica
    fig.update_xaxes(
        mirror=True,
        showline=True,
        linecolor='green',
        gridcolor='gray',
        showgrid=False
    )
    fig.update_yaxes(
        mirror=True,
        showline=True,
        linecolor='green',
        gridcolor='gray',
        showgrid=False
    )

    return fig


# Modelo Lotka-Volterra
def ecuacion_lv(a:float,b:float,c:float,d:float,P0:float,D0:float,t:float,t_i:float,cant:float):

    # Estado inicial del sistema, usado para calcular numericamente las
    # soluciones
    estado_inicial = [P0,D0]
    P_values = np.linspace(0, 500, cant)
    t_values = np.linspace(t_i, t, cant)

    # Crear una malla de puntos (P, t)
    T, P = np.meshgrid(t_values, P_values)

    # Definimos el sistema
    def lotka_volterra(est_ini,t,a,b,c,d):
        P,D = est_ini
        dP = a*P - b*P*D
        dD = - c*D + d*P*D
        return [dP,dD]

    # Encontramos numericamente las soluciones al sistema usando scipy
    sol = odeint(lotka_volterra,estado_inicial,t_values,args=(a,b,c,d))

    # Almacenamos las soluciones para ploteo
    presas, depredadores = sol[:,0], sol[:,1]

    # Ploteo
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x = t_values,
            y = presas,
            #mode = 'markers+lines',
            line=dict(color='blue'),
            name = 'Poblacion presas',
            line_shape='spline'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x = t_values,
            y = depredadores,
            #mode = 'markers+lines',
            line=dict(color='red'),
            name = 'Poblacion depredadores',
            line_shape='spline'

        )
    )


    # Etiquetas para la gráfica
    fig.update_layout(
        title={
            'text':'Comportamiento poblacion depredador presa',
            'x':0.5,
            'y':0.92,
            'xanchor':'center'
        },
        xaxis_title='Tiempo (t)',
        yaxis_title='Población (P)',
        width=800,
        template='ggplot2',
        margin=dict(l=10,r=10,t=90,b=0),
        paper_bgcolor='#f3eddf',
        legend=dict(orientation='h',y=1.1)
    )

    # contorno a la grafica
    fig.update_xaxes(
        mirror=True,
        showline=True,
        linecolor='green',
        gridcolor='gray',
        showgrid=False
    )
    fig.update_yaxes(
        mirror=True,
        showline=True,
        linecolor='green',
        gridcolor='gray',
        showgrid=False,
    )

    return fig

