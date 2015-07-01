import numpy as np
import matplotlib.pyplot as plt

"""
Programa basado en el siguiente articulo:

[1]
Mahon, Keith I. 
"The New “York” regression: Application of an improved statistical method to geochemistry." 

"""

"""
Primero definimos las funciones que necesitamos. Al final de todas
las funciones, empieza la ejecucion del programa.
"""

def leer_entrada():
    """
    Esta funcion simplemente lee los puntos introducidos por el
    usuario.
    """
    X = list()
    Y = list()
#    Wx = list()
#    Wy = list()

    puntos = int(input("Cuantos puntos vas a introducir? "))
    i = 0
    while i < puntos:
        X.append(float(input("x"+str(i+1)+": ")))
#        Wx.append(float(input("Peso x"+str(i+1)+": ")))
        Y.append(float(input("y"+str(i+1)+": ")))
#        Wy.append(float(input("Peso y"+str(i+1)+": ")))
        i+=1
    return (X,Y)
#    return (X,Wx,Y,Wy)


def _calcular_W(b,xerr,yerr,r):
    """
    Esta funcion calcula el vector W, que esta definido en [1] (el articulo),
    justo despues de la ecuacion (7), pagina 3.
    """
    var_x = xerr**2
    var_y = yerr**2

    return (var_y+b**2*var_x-2*b*r*xerr*yerr)**-1

def _calcular_barra(X,W):
    """
    Esta funcion calcula los promedios ponderados de X y Y,
    conocidos como X barra y Y barra. Definidos en [1], justo
    despues de la ecuacion (9), pagina 3.
    """
    sW = sum(W)
    x_barra = sum(W*X)/sW
    return x_barra

def _iteraciones(X,Y,pondx,pondy,xerr,yerr, r,
                 max_i,tol,b=0.0):
    """
    Esta es la funcion principal, encargada de calcular la pendiente (b)
    y la ordenada al origen (a).
    """
    iteraciones = 0
    prev_b = np.Inf # Al inicio, b_anterior es Infinito
    converg = False

    while(iteraciones < max_i):
        # Si el cambio es b es menor a la tolerancia de error, aceptamos
        # la solucion actual
        if abs(prev_b - b) < tol:
            converg = True
            break
        prev_b = b

        W = _calcular_W(b,xerr,yerr,r)
        X_barra = _calcular_barra(X,W)
        Y_barra = _calcular_barra(Y,W)
        U = X - X_barra
        V = Y - Y_barra

        arriba = sum(W**2 * V * (U * yerr + b * V * xerr - r * V * xerr * yerr))
        abajo =  sum(W**2 * U * (U * yerr + b * V * xerr - b * r * U * xerr * yerr))

        b = arriba/abajo

        iteraciones += 1

    # Si salimos del ciclo sin haber convergencia, devolvemos los
    # valores que obtuvimos, avisando que no hubo convergencia
    X_barra = _calcular_barra(X,W)
    Y_barra = _calcular_barra(Y,W)
    a = Y_barra - b*X_barra
    return b,a, converg

def regresion_ny(X,Y,
                 pondx = None,pondy = None,
                 r = 0,
                 max_iteraciones = 500,
                 tolerancia = 1e-15):
    """
    Esta funcion recibe X y Y en forma de arreglos de numpy
    y devuelve (a, b); que son la pendiente y la ordenada
    al origen.
    Entradas:
    - X - Lista de puntos en X
    - Y - Lista de puntos en Y
    - pondx, pondy - Ponderaciones de los puntos en X y en Y
    - max_iteraciones - El numero maximo de iteraciones
    - tolerancia - El valor maximo de cambio de b para dejar de iterar
    - r - Son los coeficientes de correlacion entre los errores de X y Y.
        - asumimos que son 0.
    """
    if pondx is None: # Si no recibimos las ponderaciones, las hacemos 1
        pondx = np.asarray([1 for i in X])
    if pondy is None:
        pondy = np.asarray([1 for i in Y])

    xerr = 1/pondx**0.5
    yerr = 1/pondy**0.5

    return _iteraciones(X,Y,pondx,pondy,xerr=xerr, yerr=yerr, r=r,
                        max_i = max_iteraciones,tol = tolerancia)

def graficar_puntos(X,Y,pend,ord_origen):
    """
    Esta funcion toma como entrada los puntos X, Y y los datos
    sobre la regresion lineal; y grafica los puntos, asi como
    la linea que resulta de la regresion.
    """
    plt.scatter(X,Y,label="Datos de entrada")
    lim = plt.xlim()
    ylim = plt.ylim()
    plt.plot(np.asarray(lim),np.asarray(lim)*pend+ord_origen,
             c="red",
             label = "Resultado de la regresion")
    plt.xlim(lim)
    plt.ylim(ylim)
    plt.title("Regresion lineal \"new\" York")
    plt.ylabel("Variable dependiente")
    plt.xlabel("Variable independiente")
    plt.legend(loc='lower right')
    plt.savefig("grafica.png")


"""
Aqui empieza la ejecucion del programa
"""

"""
Estos son datos de prueba. Para ver esta prueba, se puede comentar la linea
que contiene "(X, Y) = leer_entrada()"
"""
X = [0, 0.9, 1.8, 2.6, 3.3, 4.4, 5.2, 6.1, 6.5, 7.4]
Y = [5.9, 5.4, 4.4, 4.6, 3.5, 3.7, 2.8, 2.8, 2.4, 1.5]

X = np.asarray(X)
Y = np.asarray(Y)

(X,Y) = leer_entrada()

res = regresion_ny(X,Y)
pend = res[0]
ord_origen = res[1]
converg = res[2]

if converg:
    print("El algoritmo tuvo convergencia para el modelo. Y = B*x + A")
if not converg:
    print("El algoritmo no convergio.")
print("A = "+str(ord_origen)+" | B = "+str(pend))

graficar_puntos(X,Y,pend,ord_origen)
