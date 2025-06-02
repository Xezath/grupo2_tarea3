
import math
import random
import csv


#Paso 1: Lectura de instancia
def leer_instancia_tsp(ruta_archivo):
    # Abre el archivo con la ruta especificada
    with open(ruta_archivo, 'r') as f:
        # Lee el número de ciudades
        num_ciudades = int(f.readline())
        # Lee el número de objetivos (esperamos 2)
        num_objetivos = int(f.readline())
        
        # Leer la matriz de distancias para el primer objetivo
        matriz1 = []
        for _ in range(num_ciudades):
            fila = list(map(float, f.readline().split()))
            matriz1.append(fila)
        
        # Leer línea en blanco como separador
        f.readline()
        
        # Leer la matriz de distancias para el segundo objetivo
        matriz2 = []
        for _ in range(num_ciudades):
            fila = list(map(float, f.readline().split()))
            matriz2.append(fila)
    
    # Retorna el número de ciudades y ambas matrices de distancias
    return num_ciudades, matriz1, matriz2


# Paso 2: Calcular costos (objetivos) de una ruta
def calcular_costos(ruta, matriz1, matriz2):
    # Calcula la suma de distancias para el primer objetivo
    costo1 = sum(matriz1[ruta[i]][ruta[i+1]] for i in range(len(ruta)-1))
    costo1 += matriz1[ruta[-1]][ruta[0]]  # Cierra el ciclo volviendo a la ciudad inicial
    
    # Calcula la suma de distancias para el segundo objetivo
    costo2 = sum(matriz2[ruta[i]][ruta[i+1]] for i in range(len(ruta)-1))
    costo2 += matriz2[ruta[-1]][ruta[0]]
    
    return costo1, costo2


#Paso 3: Generar población inicial
# Genera una única ruta aleatoria (permuta de las ciudades)
def generar_ruta_aleatoria(num_ciudades):
    ruta = list(range(num_ciudades))
    random.shuffle(ruta)
    return ruta

# Genera una población de rutas aleatorias
def generar_poblacion_inicial(num_ciudades, tam_poblacion):
    return [generar_ruta_aleatoria(num_ciudades) for _ in range(tam_poblacion)]


#Paso 4: Funciones de dominancia y no dominancia

# Verifica si la solución cost1 domina a cost2
def domina(cost1, cost2):
    return (cost1[0] <= cost2[0] and cost1[1] <= cost2[1]) and (cost1[0] < cost2[0] or cost1[1] < cost2[1])

# Calcula los frentes de Pareto de un conjunto de soluciones
def calcular_frentes(costos):
    frentes = []
    S = [[] for _ in range(len(costos))]    # Dominados por cada solución
    n = [0 for _ in range(len(costos))]     # Número de soluciones que dominan a cada solución
    rank = [0 for _ in range(len(costos))]
    front = []

    # Compara cada par de soluciones para determinar relaciones de dominancia
    for p in range(len(costos)):
        S[p] = []
        n[p] = 0
        for q in range(len(costos)):
            if domina(costos[p], costos[q]):
                S[p].append(q)
            elif domina(costos[q], costos[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            front.append(p)
    frentes.append(front)

    # Genera frentes sucesivos
    i = 0
    while len(frentes[i]) > 0:
        Q = []
        for p in frentes[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        frentes.append(Q)
    frentes.pop()  # Elimina el último frente vacío
    return frentes


#Paso 6: Operadores genéticos (crossover y mutación)

# Cruce tipo Order Crossover (OX)
def crossover_OX(parent1, parent2):
    size = len(parent1)
    # Selecciona dos puntos aleatorios para el cruce
    start, end = sorted(random.sample(range(size), 2))
    child = [None]*size
    # Copia el segmento del primer padre
    child[start:end+1] = parent1[start:end+1]

    # Rellena el resto con genes del segundo padre, en orden, sin duplicados
    fill_pos = (end + 1) % size
    parent2_pos = (end + 1) % size
    while None in child:
        if parent2[parent2_pos] not in child:
            child[fill_pos] = parent2[parent2_pos]
            fill_pos = (fill_pos + 1) % size
        parent2_pos = (parent2_pos + 1) % size
    return child

# Mutación por intercambio (swap)
def mutacion_swap(ruta, prob_mut=0.1):
    ruta = ruta.copy()
    if random.random() < prob_mut:
        i, j = random.sample(range(len(ruta)), 2)
        ruta[i], ruta[j] = ruta[j], ruta[i]
    return ruta


# Paso 7: Selección por torneo basada en ranking y crowding distance
def seleccion_torneo(poblacion, costos, ranks, distancias, k=2):
    seleccionados = []
    n = len(poblacion)
    k = min(k, n)  # Asegura que k no exceda el tamaño de la población
    for _ in range(n):
        if n == 1:
            seleccionados.append(poblacion[0])
            continue
        # Selecciona k candidatos al azar
        candidatos = random.sample(range(n), k)
        mejor = candidatos[0]
        # Compara por ranking y luego por crowding distance
        for c in candidatos[1:]:
            if ranks[c] < ranks[mejor]:
                mejor = c
            elif ranks[c] == ranks[mejor]:
                if distancias[c] > distancias[mejor]:
                    mejor = c
        seleccionados.append(poblacion[mejor])
    return seleccionados


#PASO 11: Métricas M1, M2, M3, Error

# Distancia euclidiana entre dos puntos (2D)
def distancia_euclidiana(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# M1: promedio de distancias del frente obtenido al frente ideal
def evaluar_M1(frente_algo, ytrue):
    return sum(min(distancia_euclidiana(a, y) for y in ytrue) for a in frente_algo) / len(frente_algo)

# M2: promedio de distancias del frente ideal al frente obtenido
def evaluar_M2(frente_algo, ytrue):
    return sum(min(distancia_euclidiana(y, a) for a in frente_algo) for y in ytrue) / len(ytrue)

# M3: dispersión del frente obtenido (uniformidad)
def evaluar_M3(frente_algo):
    if len(frente_algo) <= 1:
        return 0
    # Ordenar por f1 para medir dispersión
    frente_ordenado = sorted(frente_algo, key=lambda x: x[0])
    distancias = [distancia_euclidiana(frente_ordenado[i], frente_ordenado[i+1]) for i in range(len(frente_ordenado)-1)]
    return sum(distancias) / len(distancias)

# Error: proporción de soluciones del frente ideal no encontradas
def evaluar_error(frente_algo, ytrue):
    ytrue_set = set(tuple(p) for p in ytrue)
    frente_set = set(tuple(p) for p in frente_algo)
    no_encontradas = ytrue_set - frente_set
    return len(no_encontradas) / len(ytrue)


# Paso 10: Construcción del frente ideal (Ytrue) a partir de múltiples ejecuciones
def construir_frente_Ytrue(lista_frentes):
    # Junta todos los frentes en una sola lista
    todos_costos = [costo for frente in lista_frentes for costo in frente]

    # Filtrar solo las soluciones no dominadas
    ytrue = []
    for i, ci in enumerate(todos_costos):
        dominado = False
        for j, cj in enumerate(todos_costos):
            if i != j and domina(cj, ci):
                dominado = True
                break
        if not dominado:
            ytrue.append(ci)
    return ytrue