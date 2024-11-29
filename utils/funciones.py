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
        legend=dict(orientation='h',y=1.2)
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
        legend=dict(orientation='h',y=1.2),
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
        legend=dict(orientation='h',y=1.2)
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
        legend=dict(orientation='h',y=1.2)
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


def modelo_sir(N, I0, R0, beta, gamma, tiempo_total, n, t_variacion,beta_,gamma_):
    
    def sir(y, t, N, beta, gamma):
        S, I, R = y
        if t >= t_variacion:
            beta = beta_
            gamma = gamma_
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    S0 = N - I0 - R0
    y0 = S0, I0, R0
    t = np.linspace(0, tiempo_total, n)
    
    solutions = odeint(sir, y0, t, args=(N, beta, gamma))
    S, I, R = solutions.T

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=S, mode='lines', name='Susceptibles', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=t, y=I, mode='lines', name='Infectados', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=t, y=R, mode='lines', name='Recuperados', line=dict(color='green')))
    
    # Etiquetas para la gráfica
    fig.update_layout(
        title={
            'text':'Modelo SIR',
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
        legend=dict(orientation='h',y=1.2)
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


def modelo_lv_comp_intra(X0, Y0, Z0, a21, a22, a23, a31, a32, a33, d1, d2, t, cant):
    
    estado_inicial = [X0,Y0,Z0]
    P_values = np.linspace(0, 1, cant)
    t_values = np.linspace(0, t, cant)

    # Crear una malla de puntos (P, t)
    T, P = np.meshgrid(t_values, P_values)

    # Definimos el sistema
    def lotka_volterra(est_ini,t,a21,a22,a23,a31,a32,a33,d1,d2):
        X,Y,Z = est_ini
        dX = X*(1 - X - Y - Z)
        dY = Y*(-d1 + a21*X - a22*Y - a23*Z)
        dZ = Z*(-d2 + a31*X + a32*Y - a33*Z)
        return dX, dY, dZ

    # Encontramos numericamente las soluciones al sistema usando scipy
    sol = odeint(lotka_volterra,estado_inicial,t_values,args=(a21,a22,a23,a31,a32,a33,d1,d2))

    # Almacenamos las soluciones para ploteo
    presas, depredadores_int, depredadores_sup = sol[:,0], sol[:,1], sol[:,2]

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
            y = depredadores_int,
            #mode = 'markers+lines',
            line=dict(color='red'),
            name = 'Poblacion depredadores intermedio',
            line_shape='spline'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x = t_values,
            y = depredadores_sup,
            #mode = 'markers+lines',
            line=dict(color='green'),
            name = 'Poblacion depredadores superior',
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
        legend=dict(orientation='h',y=1.2)
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


def modelo_lv_comp_intra_3d(X0, Y0, Z0, a21, a22, a23, a31, a32, a33, d1, d2, t, cant):
    
    estado_inicial = [X0,Y0,Z0]
    P_values = np.linspace(0, 1, cant)
    t_values = np.linspace(0, t, cant)

    # Crear una malla de puntos (P, t)
    T, P = np.meshgrid(t_values, P_values)

    # Definimos el sistema
    def lotka_volterra(est_ini,t,a21,a22,a23,a31,a32,a33,d1,d2):
        X,Y,Z = est_ini
        dX = X*(1 - X - Y - Z)
        dY = Y*(-d1 + a21*X - a22*Y - a23*Z)
        dZ = Z*(-d2 + a31*X + a32*Y - a33*Z)
        return dX, dY, dZ

    # Encontramos numericamente las soluciones al sistema usando scipy
    sol = odeint(lotka_volterra,estado_inicial,t_values,args=(a21,a22,a23,a31,a32,a33,d1,d2))

    # Almacenamos las soluciones para ploteo
    presas, depredadores_int, depredadores_sup = sol[:,0], sol[:,1], sol[:,2]

    # Ploteo
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x = presas,
            y = depredadores_int,
            z = depredadores_sup,
            #mode = 'markers+lines',
            line=dict(color='blue'),
            marker=dict(
                size=12,
                color=t_values,                # set color to an array/list of desired values
                colorscale='Viridis',   # choose a colorscale
                opacity=0.8
            ),
            name = 'Poblacion presas',
            # line_shape='spline'
        )
    )

    fig.update_layout(
        scene = dict(
            xaxis = dict(nticks=4, range=[0,1], title=dict(text="Poblacion de Presa")),
            yaxis = dict(nticks=4, range=[0,1], title=dict(text="Poblacion de Depredador Intermedio")),
            zaxis = dict(nticks=4, range=[0,1], title=dict(text="Poblacion de Depredador Superior")),),
        title={
            'text':'Comportamiento poblacion depredador presa',
            'x':0.5,
            'y':0.92,
            'xanchor':'center'
        },
        width=800,
        template='ggplot2',
        # margin=dict(l=10,r=10,t=90,b=0),
        paper_bgcolor='#f3eddf',
        legend=dict(orientation='h',y=1.2),
        margin=dict(r=20, l=10, b=10, t=10))

    # Etiquetas para la gráfica
    # fig.update_layout(
    #     title={
    #         'text':'Comportamiento poblacion depredador presa',
    #         'x':0.5,
    #         'y':0.92,
    #         'xanchor':'center'
    #     },
    #     xaxis_title='Presas',
    #     yaxis_title='Depr. Intermedio',
    #     # zaxis_title='Depr. Superior',
    #     width=800,
    #     template='ggplot2',
    #     margin=dict(l=10,r=10,t=90,b=0),
    #     paper_bgcolor='#f3eddf',
    #     legend=dict(orientation='h',y=1.2)
    # )

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
    # fig.update_zaxes(
    #     mirror=True,
    #     showline=True,
    #     linecolor='green',
    #     gridcolor='gray',
    #     showgrid=False,
    # )

    return fig
