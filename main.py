import csv
import os
import matplotlib.pyplot as plt
import random

from TSP_bi_Objetivo import evaluar_algoritmo_nsga, evaluar_algoritmo_spea
from utils import construir_frente_Ytrue, evaluar_M1, evaluar_M2, evaluar_M3, evaluar_error

def guardar_métricas_csv(resultados, archivo_salida="resultados_metricas.csv"):
    campos = ["Algoritmo", "Instancia", "M1", "M2", "M3", "Error"]
    with open(archivo_salida, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for r in resultados:
            writer.writerow(r)

def guardar_frente_csv(frente, nombre_archivo):
    with open(nombre_archivo, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Objetivo 1", "Objetivo 2"])
        for costo in frente:
            writer.writerow(costo)

def graficar_frentes_comparados(frente_nsga, frente_spea, nombre_instancia):
    x_nsga = [c[0] for c in frente_nsga]
    y_nsga = [c[1] for c in frente_nsga]

    x_spea = [c[0] for c in frente_spea]
    y_spea = [c[1] for c in frente_spea]

    plt.figure()
    plt.scatter(x_nsga, y_nsga, color='blue', label='NSGA-II', alpha=0.6)
    plt.scatter(x_spea, y_spea, color='red', label='SPEA', alpha=0.6)
    plt.title(f"Comparación de Frentes - {nombre_instancia}")
    plt.xlabel("Objetivo 1")
    plt.ylabel("Objetivo 2")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"comparacion_frentes_{nombre_instancia}.png")
    plt.close()

def main():
    random.seed(42)
    instancias = {
        "KROAB100": "tsp_KROAB100.TSP.TXT",
        "KROAC100": "tsp_kroac100.tsp.txt"
    }

    resultados = []

    for nombre, path in instancias.items():
        print(f"--- Instancia: {nombre} ---")

        print("Ejecutando NSGA-II...")
        res_nsga = evaluar_algoritmo_nsga(path)

        print("Ejecutando SPEA...")
        res_spea = evaluar_algoritmo_spea(path)

        # Construir Ytrue combinando todos los frentes de ambas ejecuciones
        todos_frentes = res_nsga["Frentes"] + res_spea["Frentes"]
        ytrue = construir_frente_Ytrue(todos_frentes)

        # Recalcular métricas para NSGA-II
        m1_nsga = evaluar_M1(res_nsga["Frentes"][0], ytrue)
        m2_nsga = evaluar_M2(res_nsga["Frentes"][0], ytrue)
        m3_nsga = evaluar_M3(res_nsga["Frentes"][0])
        err_nsga = evaluar_error(res_nsga["Frentes"][0], ytrue)

        # Recalcular métricas para SPEA
        m1_spea = evaluar_M1(res_spea["Frentes"][0], ytrue)
        m2_spea = evaluar_M2(res_spea["Frentes"][0], ytrue)
        m3_spea = evaluar_M3(res_spea["Frentes"][0])
        err_spea = evaluar_error(res_spea["Frentes"][0], ytrue)

        # Guardar CSVs de frentes
        guardar_frente_csv(res_nsga["Frentes"][0], f"frente_nsga_{nombre}.csv")
        guardar_frente_csv(res_spea["Frentes"][0], f"frente_spea_{nombre}.csv")

        # Guardar resultados de métricas
        resultados.append({
            "Algoritmo": "NSGA-II",
            "Instancia": nombre,
            "M1": m1_nsga,
            "M2": m2_nsga,
            "M3": m3_nsga,
            "Error": err_nsga
        })

        resultados.append({
            "Algoritmo": "SPEA",
            "Instancia": nombre,
            "M1": m1_spea,
            "M2": m2_spea,
            "M3": m3_spea,
            "Error": err_spea
        })

        # Graficar comparación de frentes
        graficar_frentes_comparados(res_nsga["Frentes"][0], res_spea["Frentes"][0], nombre)

    # Guardar resultados finales de métricas
    guardar_métricas_csv(resultados)
    print("✅ Resultados guardados en 'resultados_metricas.csv'")


if __name__ == "__main__":
    main()
