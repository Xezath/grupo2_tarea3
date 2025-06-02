import utils
from utils import (    evaluar_M1,
    evaluar_M2,
    evaluar_M3,
    evaluar_error,
    construir_frente_Ytrue,
    leer_instancia_tsp,
    calcular_costos,    
    generar_poblacion_inicial,
    calcular_frentes,
    domina,
    generar_ruta_aleatoria,
    crossover_OX,   
    mutacion_swap,
    seleccion_torneo

)

import math
import random
random.seed(42)  # Semilla fija para reproducibilidad
import csv
import os    



#Paso 5: Distancia de hacinamiento (crowding distance)
def distancia_hacinamiento(costos, frente):
    n = len(frente) # Número de individuos en el frente
    dist = [0.0] * n    # Inicializamos todas las distancias en cero
    for m in range(2):  # Iteramos sobre los dos objetivos
        # Obtenemos los valores del objetivo m para los individuos del frente
        valores = [costos[i][m] for i in frente]
        # Ordenamos los índices del frente según el valor del objetivo m
        orden = sorted(range(n), key=lambda idx: costos[frente[idx]][m])
        # Asignamos infinita distancia a los extremos para conservar la diversidad
        dist[orden[0]] = float('inf')
        dist[orden[-1]] = float('inf')
        max_val = valores[orden[-1]]
        min_val = valores[orden[0]]
        # Evitamos división por cero
        if max_val == min_val:
            continue
        # Calculamos la distancia normalizada para los individuos intermedios
        for i in range(1, n-1):
            dist[orden[i]] += (valores[orden[i+1]] - valores[orden[i-1]]) / (max_val - min_val)
    return dist # Devuelve un vector con la distancia de hacinamiento para cada individuo del frente


#Paso 8: NSGA-II completo (simplificado)
def nsga2(num_ciudades, matriz1, matriz2, tam_poblacion=150, generaciones=100, prob_mutacion=0.2):
    # Generamos una población inicial de rutas aleatorias
    poblacion = generar_poblacion_inicial(num_ciudades, tam_poblacion)

    for gen in range(generaciones): # Iteramos sobre cada generación
        # Calculamos los costos (2 objetivos) de cada individuo
        costos = [calcular_costos(ind, matriz1, matriz2) for ind in poblacion]
        # Calculamos los frentes de Pareto usando dominancia
        frentes = calcular_frentes(costos)

        nueva_poblacion = []
        for frente in frentes:
            # Calculamos la distancia de hacinamiento para cada frente
            dist = distancia_hacinamiento(costos, frente)
            # Ordenamos el frente en base a la distancia (más diversidad primero)
            orden_frente = sorted(zip(frente, dist), key=lambda x: -x[1])  # solo crowding distance
            # Agregamos individuos al nuevo conjunto
            for i, _ in orden_frente:
                nueva_poblacion.append(poblacion[i])
            # Si ya llenamos la población, detenemos el proceso
            if len(nueva_poblacion) >= tam_poblacion:
                nueva_poblacion = nueva_poblacion[:tam_poblacion]
                break


        # Recalculamos costos y frentes para la nueva población
        costos = [calcular_costos(ind, matriz1, matriz2) for ind in nueva_poblacion]
        frentes = calcular_frentes(costos)

        # Calculamos distancias de hacinamiento para selección por torneo
        distancias = []
        for frente in frentes:
            distancias += distancia_hacinamiento(costos, frente)

        # Asignamos rango (nivel de frente) a cada individuo
        ranks = [0]*len(nueva_poblacion)
        for i, frente in enumerate(frentes):
            for ind in frente:
                ranks[ind] = i

        # Seleccionamos padres usando torneo basado en rango y hacinamiento 
        seleccion = seleccion_torneo(nueva_poblacion, costos, ranks, distancias)

        # Aplicamos cruzamiento y mutación para crear la siguiente generación
        siguiente_poblacion = []
        for i in range(0, tam_poblacion, 2):
            padre1 = seleccion[i]
            padre2 = seleccion[(i+1) % tam_poblacion]
            hijo1 = crossover_OX(padre1, padre2)
            hijo2 = crossover_OX(padre2, padre1)
            hijo1 = mutacion_swap(hijo1, prob_mutacion)
            hijo2 = mutacion_swap(hijo2, prob_mutacion)
            siguiente_poblacion.extend([hijo1, hijo2])

        poblacion = siguiente_poblacion[:tam_poblacion] # Actualizamos población

    # Al final, devolver el conjunto Pareto de la última generación
    costos_final = [calcular_costos(ind, matriz1, matriz2) for ind in poblacion]
    frentes_finales = calcular_frentes(costos_final)
    frente_pareto = [poblacion[i] for i in frentes_finales[0]]
    costos_pareto = [costos_final[i] for i in frentes_finales[0]]
    return frente_pareto, costos_pareto


#PASO 9: Ejecutar NSGA-II 5 veces y guardar los resultados
def ejecutar_nsga_varias_veces(instancia_path, repeticiones=5):
    num_ciudades, matriz1, matriz2 = leer_instancia_tsp(instancia_path)
    frentes_todas = []

    for i in range(repeticiones):
        random.seed(42 + i) # Cambiamos la semilla en cada repetición
        print(f"Ejecutando corrida {i+1}...")
        frente_pareto, costos_pareto = nsga2(num_ciudades, matriz1, matriz2, 150, 100)
        frentes_todas.append(costos_pareto)

    return frentes_todas


#PASO 12: Unir todo
#Función para obtener métricas promediadas de 5 ejecuciones:
def evaluar_algoritmo_nsga(instancia_path, repeticiones=5):
    frentes_algo = ejecutar_nsga_varias_veces(instancia_path, repeticiones)
    frente_completo = [c for frente in frentes_algo for c in frente]
    ytrue = construir_frente_Ytrue(frentes_algo)

    m1s, m2s, m3s, errors = [], [], [], []

    for frente in frentes_algo:
        m1s.append(evaluar_M1(frente, ytrue))       # Cobertura
        m2s.append(evaluar_M2(frente, ytrue))       # Uniformidad
        m3s.append(evaluar_M3(frente))              # Dispersión
        errors.append(evaluar_error(frente, ytrue)) # Error global

    # Retorna los promedios de las métricas y datos relevantes
    return {
        "M1": sum(m1s) / repeticiones,
        "M2": sum(m2s) / repeticiones,
        "M3": sum(m3s) / repeticiones,
        "Error": sum(errors) / repeticiones,
        "Ytrue": ytrue,
        "Frentes": frentes_algo
    }

#SPEA para TSP Bi-objetivo
#Paso 1: Fuerza (strength) y aptitud (fitness)
def calcular_strengths(costos):
    n = len(costos)
    strengths = [0] * n
    for i in range(n):
        for j in range(n):
            if domina(costos[i], costos[j]):
                strengths[i] += 1
    return strengths

def calcular_fitness(costos, strengths):
    n = len(costos)
    fitness = [0] * n
    for i in range(n):
        for j in range(n):
            if domina(costos[j], costos[i]):
                fitness[i] += strengths[j]
    return fitness


#Paso 2: Filtrado de no dominados
def filtrar_no_dominados(costos, poblacion):
    no_dominados = []
    for i, ci in enumerate(costos):
        es_dominado = False
        for j, cj in enumerate(costos):
            if i != j and domina(cj, ci):
                es_dominado = True
                break
        if not es_dominado:
            no_dominados.append((poblacion[i], ci))
    return no_dominados


#Paso 3: SPEA completo
def spea(num_ciudades, matriz1, matriz2, tam_poblacion=150, tamano_archivo=75, generaciones=100, prob_mutacion=0.2):
    poblacion = generar_poblacion_inicial(num_ciudades, tam_poblacion)
    archivo = []

    for gen in range(generaciones):
        union = archivo + poblacion
        costos_union = [calcular_costos(ind, matriz1, matriz2) for ind in union]
        
        strengths = calcular_strengths(costos_union)
        fitness = calcular_fitness(costos_union, strengths)

        # Selección de archivo: guardar no dominados hasta llenar el archivo
        candidatos = list(zip(union, costos_union, fitness))
        candidatos.sort(key=lambda x: x[2])  # ordenar por fitness
        archivo = [ind for ind, _, _ in candidatos[:tamano_archivo]]

        if len(archivo) == 0:
            archivo = random.choices(poblacion, k=tamano_archivo)

        # Selección de padres desde el archivo
        costos_archivo = [calcular_costos(ind, matriz1, matriz2) for ind in archivo]
        seleccion = []
        while len(seleccion) < tam_poblacion:
            nuevos = seleccion_torneo(archivo, costos_archivo, [0]*len(archivo), [1]*len(archivo))
            if not nuevos:
                nuevos = random.choices(archivo, k=2)  # permite duplicados si es necesario
            seleccion += nuevos

        seleccion = seleccion[:tam_poblacion]

        # Reproducir para generar nueva población
        nueva_poblacion = []
        for i in range(0, tam_poblacion, 2):
            p1 = seleccion[i % len(seleccion)]
            p2 = seleccion[(i+1) % len(seleccion)]
            h1 = crossover_OX(p1, p2)
            h2 = crossover_OX(p2, p1)
            h1 = mutacion_swap(h1, prob_mutacion)
            h2 = mutacion_swap(h2, prob_mutacion)
            nueva_poblacion.extend([h1, h2])
        
        poblacion = nueva_poblacion[:tam_poblacion]

    # Al final: devolver soluciones no dominadas del archivo final
    costos_final = [calcular_costos(ind, matriz1, matriz2) for ind in archivo]
    frentes = calcular_frentes(costos_final)
    frente_pareto = [archivo[i] for i in frentes[0]]
    costos_pareto = [costos_final[i] for i in frentes[0]]
    return frente_pareto, costos_pareto


#Ejecución Múltiple y Métricas para SPEA
def ejecutar_spea_varias_veces(instancia_path, repeticiones=5):
    num_ciudades, matriz1, matriz2 = leer_instancia_tsp(instancia_path)
    frentes_todas = []

    for i in range(repeticiones):
        random.seed(42 + i)
        print(f"Ejecutando SPEA corrida {i+1}...")
        frente_pareto, costos_pareto = spea(num_ciudades, matriz1, matriz2, 150, 75, 100)
        frentes_todas.append(costos_pareto)

    return frentes_todas

def evaluar_algoritmo_spea(instancia_path, repeticiones=5):
    frentes_algo = ejecutar_spea_varias_veces(instancia_path, repeticiones)
    frente_completo = [c for frente in frentes_algo for c in frente]
    ytrue = construir_frente_Ytrue(frentes_algo)

    m1s, m2s, m3s, errors = [], [], [], []

    for frente in frentes_algo:
        m1s.append(evaluar_M1(frente, ytrue))
        m2s.append(evaluar_M2(frente, ytrue))
        m3s.append(evaluar_M3(frente))
        errors.append(evaluar_error(frente, ytrue))

    return {
        "M1": sum(m1s) / repeticiones,
        "M2": sum(m2s) / repeticiones,
        "M3": sum(m3s) / repeticiones,
        "Error": sum(errors) / repeticiones,
        "Ytrue": ytrue,
        "Frentes": frentes_algo
    }



# # Leer instancia
# num_ciudades, matriz1, matriz2 = leer_instancia_tsp("tsp_KROAB100.TSP.TXT")

# # Ejecutar NSGA-II
# frente_pareto, costos_pareto = nsga2(num_ciudades, matriz1, matriz2, tam_poblacion=100, generaciones=100)

# # Mostrar resultados
# for ruta, costo in zip(frente_pareto, costos_pareto):
#     print(f"Costo: {costo}, Ruta: {ruta}")