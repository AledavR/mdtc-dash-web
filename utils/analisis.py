import sympy as sp
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

def format_matrix(matrix):
      matrix_latex = "\\begin{bmatrix}\n"
      for i in range(matrix.rows):
          fila = " & ".join([sp.latex(elem) for elem in matrix.row(i)])
          matrix_latex += fila
          if i < matrix.rows - 1:
              matrix_latex += " \\\\\n"
          else:
              matrix_latex += "\n"
      matrix_latex += "\\end{bmatrix}"
      return matrix_latex

def clas_punto(trace,det):
    if det < 0:
        return "punto silla"
    elif det > 0:
        disc = (trace**2) - (4*det)
        if disc < 0:
            if trace < 0:
                return  "foco estable"
            elif trace > 0:
                return "foco inestable"
            elif trace == 0:
                return "vortice"
        elif disc > 0:
            if trace < 0:
                return  "nodo estable"
            elif trace > 0:
                return "nodo inestable"
        elif disc == 0:
            return  "nodo degenerado"

def analisis_punto_critico(equations, symbols, crit_points):
    x,y = symbols
    jacobian = sp.simplify(equations.jacobian(symbols))
    latex_jacobian = format_matrix(jacobian)

    points_analysis = ""

    for point in crit_points:
        jacobian_eval = jacobian.subs({x: point[0], y:point[1]})
        trace = jacobian_eval.trace()
        det = jacobian_eval.det()
        points_analysis += f"El punto {point} es un {clas_punto(trace,det)}\n"
    return latex_jacobian, points_analysis

def analisis_ode(dxdt, dydt,min,max, n_clicks):
    if n_clicks == 0:
        return "Ingrese el sistema a analizar:","","",go.Figure()

    try:
        x,y = sp.symbols('x y')
        x_,y_ = sp.sympify(dxdt), sp.sympify(dydt)
        crit_points = np.array(sp.solve([x_, y_], (x, y))).astype(float)
        res_points = f'Los puntos criticos son: {crit_points}'
        jacobiano, analisis_puntos = analisis_punto_critico(sp.Matrix([x_,y_]),[x,y],crit_points)

        x_vals = np.linspace(min, max, 10) 
        y_vals = np.linspace(min, max, 10)
        X,Y = np.meshgrid(x_vals,y_vals)
        
        x_func = sp.lambdify((x, y), x_, 'numpy')
        y_func = sp.lambdify((x, y), y_, 'numpy')
        U = x_func(X, Y)
        V = y_func(X, Y)
        
     
        fig = go.Figure()
        quiver = ff.create_quiver(X, Y, U, V, name="Desarrollo del sistema")
        fig.add_traces(quiver.data)
        fig.add_traces(go.Scatter(x = crit_points[:,0], y = crit_points[:,1], mode='markers', name="Punto crítico"))
        fig.update_layout(
            title={
                'text': '"Desarrollo del sistema"',
                'x': 0.5,
                'y': 0.92,
                'xanchor': 'center'
            },
            xaxis_title='X',
            yaxis_title='Y',
            width=800,
            margin=dict(l=10, r=10, t=90, b=0),
            template='ggplot2',
            paper_bgcolor='#f3eddf',
            legend=dict(orientation='h', y=1.2)
        )
        return res_points, f"$$ \\text{{Matriz Jacobiana}} = {jacobiano} $$", analisis_puntos, fig
    
    except Exception as e:
        return f'Error al procesar las ecuaciones: {str(e)}', '', '', go.Figure()
