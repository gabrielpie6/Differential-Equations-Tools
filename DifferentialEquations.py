import numpy as np

#   Biblioteca que implementa funcionalidades para resolver 
#   sistemas de N equações diferenciais numericamente

def Diferencial(Equations:dict, Values:list, i):
    return list(Equations.values())[i](*Values)

# Aproximação Linear
def MetodoNumerico(m, x, x_0, y_0):
    return m * (x - x_0) + y_0

# Método Heun genérico
# g: função dydx
def General_Heun(g, xi, yi, h):
    k1 = g(xi, yi)
    k2 = g(xi + h, yi + k1 * h)
    k = (k1 + k2)/2
    return yi + k * h



def SolveEDO_Euler(Equations:dict, InitialValues:dict, X_final, passo):
    """
    DESCRIPTION: Resolve um sistema de N equações diferenciais ordinárias.

    PARAMETERS:
        Equations:dict - dicionário bem formatado (formato especificado em exemplos) contendo
        N equações diferenciais ordinárias;

        InitialValues:dict - dicionário bem formatado (formato especificado em exemplos) contendo N
        valores iniciais
        
        X_final - limite superior para cálculo do sistema

        passo - tamanho da partição, isto é, sendo x a variável independente (dy/dx),
        então o passo é dado pela distância entre as coordenadas x de um ponto e o seu próximo
    
    RETURN: retorna uma lista contendo um conjunto de N listas que representam os resultados para cada
    função, seguindo a ordem de boa formatação

    EXAMPLES:
        Equacoes = {
            "dL": lambda L, M, A: (r1 * M *(1 - L/Kl) - (b1 + Lambda) * L),
            "dM": lambda L, M, A: (Lambda * ((A**2) / (c1**2 + A**2)) * L * (1 - M/Km) - (b2 * M)),
            "dA": lambda L, M, A: (r2 * A * (1 - A/Ka) - b3 * ((A**2) / (c2**2 + A**2)) * M)
        }
        #
        #
        Inicial = {
            "L": 100000000,
            "M": 0,
            "A": 10000000
        }
        #
    OBSERVATIONS: Toda as funções (L, M e A) são parâmetros de 'lambda' em todas as funções e DEVEM estar sempre
    na mesma ordem, bem como na ordem de disposição das equações dentro do dicionário (dL, dM e dA) e também
    no dicionário de valores iniciais (L, M e A), esta é a boa formatação. Qualquer disparidade entre a
    ordenadação de algum destes elementos gerará ERRO. Como por exemplo: 
        Equacoes = {"dL": lambda L, M, A: ..., "dM": lambda L, M, A: ..., "dA": lambda L, M, A: ...)
        Inicial = {"M": ..., "L": ..., "A": ...}
                -> Haverá ERRO, pois a ordem foi trocada em Inicial, onde deveria ser "L", "M", "A"
    - TODA equação deve ser inserida no formato:
        "{nome}": lambda {funções bem ordenadas separadas por vírgula}: {expressão matemática da equação},
    - Somente a ÚLTIMA equação do dicionário não deve possuir vírgula ao final de {expressão matemática da equação}, ademais
    TODAS devem possuir, como indicado na formatação acima.
    """

    SYSTEM_SIZE = len(Equations)

    Values = list(InitialValues.values())
    Functions = [[Values[i]] for i in range(SYSTEM_SIZE)]
    dydx = [None] * SYSTEM_SIZE
    EixoX_List = [*np.arange(0, X_final, passo)]

    # 1 representa índice 1 da lista, e não x = 1
    for t in EixoX_List[1:]:
        # Cálculo dos Diferenciais
        for i in range(SYSTEM_SIZE):
            dydx[i] = Diferencial(Equations, Values, i)

        # Aproximação dos pontos
        for i in range(SYSTEM_SIZE):
            Values[i] = MetodoNumerico(dydx[i], t, t - passo, Values[i])
            Functions[i].append(Values[i])
    return Functions


# Método Euler Explícito para resolução uma EDO
def ExplicitEuler(diferencial, x0, xf, y0, N_Partitions):
    h = (xf - x0)/N_Partitions
    X = [x0 + h*i  for i in range(0, N_Partitions + 1)]
    Y = [y0]

    for i in range(1, N_Partitions + 1):
        Y.append(Y[i - 1] + h * diferencial(X[i - 1], Y[i - 1]))
    return (X, Y)


# Método Euler Implícito para resolução uma EDO
def ImplicitEuler(diferencial, x0, xf, y0, tol, N_Partitions, Gauss_Seidel_Iterations):
    h = (xf - x0)/N_Partitions
    X = [x0 + h*i  for i in range(0, N_Partitions + 1)]
    Y = [y0] * (N_Partitions + 1)
    y_ant = Y

    for it in range(1, Gauss_Seidel_Iterations):
        for i in range(0, N_Partitions):
            Y[i + 1] = Y[i] + h * diferencial(X[i + 1], y_ant[i + 0])

        controle = 1
        for i in range (0, N_Partitions):
            if (abs(Y[i] - y_ant[i]) > tol):
                controle = 0
        
        if (it > 1 and controle == 1):
            break
        y_ant = Y
    return (X, Y)


# Método Heun para resolução uma EDO
def Heun(diferencial, x0, xf, y0, N_Partitions):
    h = (xf - x0)/N_Partitions
    X = [x0 + h*i  for i in range(0, N_Partitions + 1)]
    Y = [y0]

    for i in range(1, N_Partitions + 1):
        Y.append(General_Heun(diferencial, X[i - 1], Y[i -1], h))
    return (X, Y)

# Método Gauss-Seidel para solucionar um sistema de equações diagonalmente dominante 
def Gauss_Seidel(Equations:dict, InitialValues:dict, error:float):
    SYSTEM_SIZE = len(Equations)
    Eq  = list(Equations.values())
    Res = list(InitialValues.values())
    Ant = list(Res)

    #repetição
    N = 0
    controle = 0
    while (controle == 0):
        controle = 1
        for i in range(0, SYSTEM_SIZE):
            Res[i] = Eq[i](*Ant)
            #print(f"abs({i}) = {abs(Res[i] - Ant[i])}")
            if (abs(Res[i] - Ant[i]) > error):
                controle = 0
        Ant = list(Res)
        N = N + 1
    N = N - 1
    #print(f"Foi preciso {N - 1} interacoes")
    return Res