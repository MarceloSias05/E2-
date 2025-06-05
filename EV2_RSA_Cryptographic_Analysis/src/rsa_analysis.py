from multiprocessing import Pool, cpu_count
from itertools import combinations
import math
import time
import os
import matplotlib.pyplot as plt
import numpy as np

def calculate_gcd_with_factorization(pair):
    """
    Calcula el GCD de un par de números y devuelve la factorización
    si hay un factor común > 1 (vulnerabilidad de primo compartido)
    """
    a, b = pair
    gcd = math.gcd(a, b)
    if gcd > 1:
        # Factorización: a/gcd y b/gcd son los otros factores
        factor_a = a // gcd
        factor_b = b // gcd
        return (a, b, gcd, factor_a, factor_b)
    return None

def read_integers_file(filename):
    """Lee los números enteros del archivo especificado"""
    try:
        with open(filename, 'r') as f:
            numbers = []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and line.isdigit():
                    numbers.append(int(line))
                elif line:  # Si hay contenido pero no es dígito
                    print(f"Advertencia: Línea {line_num} ignorada: '{line}'")
            return numbers
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{filename}'")
        return []
    except Exception as e:
        print(f"Error leyendo el archivo: {e}")
        return []

def write_results_to_file(results, filename):
    """
    Escribe los resultados en el formato requerido:
    número1, número1/gcd, gcd
    número2, número2/gcd, gcd
    """
    with open(filename, 'w') as f:
        f.write("# RSA Shared Prime Attack Results\n")
        f.write("# Format: original_number, factorized_part, shared_prime\n")
        f.write("# Each vulnerable pair generates two lines\n\n")
        
        for a, b, gcd_val, factor_a, factor_b in results:
            # Escribir ambos números del par vulnerable
            f.write(f"{a}, {factor_a}, {gcd_val}\n")
            f.write(f"{b}, {factor_b}, {gcd_val}\n")

def analyze_with_cores(numbers, num_cores, chunk_size=1000):
    """
    Realiza el análisis de primos compartidos usando un número específico de cores
    """
    print(f"\n=== Análisis con {num_cores} core(s) ===")
    
    start_time = time.time()
    
    # Generar todos los pares posibles
    total_pairs = len(numbers) * (len(numbers) - 1) // 2
    print(f"Total de pares a analizar: {total_pairs:,}")
    
    pairs_generator = combinations(numbers, 2)
    
    # Configurar pool de procesos
    vulnerable_pairs = []
    processed_count = 0
    
    if num_cores == 1:
        # Ejecución secuencial para comparación
        print("Ejecutando análisis secuencial...")
        for pair in pairs_generator:
            result = calculate_gcd_with_factorization(pair)
            processed_count += 1
            
            if result:
                vulnerable_pairs.append(result)
            
            # Progreso cada 100,000 pares
            if processed_count % 100000 == 0:
                elapsed = time.time() - start_time
                print(f"  Procesados: {processed_count:,} pares, "
                      f"Vulnerables encontrados: {len(vulnerable_pairs)}, "
                      f"Tiempo: {elapsed:.2f}s")
    else:
        # Ejecución paralela
        print(f"Ejecutando análisis paralelo con {num_cores} cores...")
        with Pool(num_cores) as pool:
            # Usar imap para procesamiento en lotes
            for result in pool.imap(calculate_gcd_with_factorization, 
                                   pairs_generator, 
                                   chunksize=chunk_size):
                processed_count += 1
                
                if result:
                    vulnerable_pairs.append(result)
                
                # Progreso cada 100,000 pares
                if processed_count % 100000 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Procesados: {processed_count:,} pares, "
                          f"Vulnerables encontrados: {len(vulnerable_pairs)}, "
                          f"Tiempo: {elapsed:.2f}s")
    
    execution_time = time.time() - start_time
    
    print(f"Análisis completado en {execution_time:.2f} segundos")
    print(f"Pares vulnerables encontrados: {len(vulnerable_pairs)}")
    
    return vulnerable_pairs, execution_time

def create_performance_graphs(timing_results):
    """
    Genera gráficas de rendimiento del análisis paralelo
    """
    cores = [result[0] for result in timing_results]
    times = [result[1] for result in timing_results]
    
    # Calcular speedup y eficiencia
    baseline_time = timing_results[0][1]  # Tiempo con 1 core
    speedups = [baseline_time / time for time in times]
    efficiencies = [speedup / core * 100 for speedup, core in zip(speedups, cores)]
    
    # Configurar el estilo de las gráficas
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('RSA Shared Prime Attack - Análisis de Rendimiento Paralelo', 
                 fontsize=16, fontweight='bold')
    
    # Gráfica 1: Cores vs Tiempo de Ejecución
    ax1.plot(cores, times, 'bo-', linewidth=2, markersize=8, label='Tiempo real')
    ax1.set_xlabel('Número de Cores')
    ax1.set_ylabel('Tiempo de Ejecución (segundos)')
    ax1.set_title('Cores vs Tiempo de Ejecución')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Agregar valores en los puntos
    for i, (core, time) in enumerate(zip(cores, times)):
        ax1.annotate(f'{time:.1f}s', (core, time), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    # Gráfica 2: Speedup vs Cores
    ax2.plot(cores, speedups, 'ro-', linewidth=2, markersize=8, label='Speedup real')
    # Línea teórica ideal (speedup = cores)
    ax2.plot(cores, cores, 'g--', linewidth=2, alpha=0.7, label='Speedup ideal')
    ax2.set_xlabel('Número de Cores')
    ax2.set_ylabel('Speedup (factor de mejora)')
    ax2.set_title('Speedup vs Número de Cores')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Agregar valores en los puntos
    for i, (core, speedup) in enumerate(zip(cores, speedups)):
        ax2.annotate(f'{speedup:.2f}x', (core, speedup), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    # Gráfica 3: Eficiencia vs Cores
    ax3.plot(cores, efficiencies, 'mo-', linewidth=2, markersize=8)
    ax3.axhline(y=100, color='g', linestyle='--', alpha=0.7, label='Eficiencia ideal (100%)')
    ax3.set_xlabel('Número de Cores')
    ax3.set_ylabel('Eficiencia (%)')
    ax3.set_title('Eficiencia de Paralelización')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Agregar valores en los puntos
    for i, (core, eff) in enumerate(zip(cores, efficiencies)):
        ax3.annotate(f'{eff:.1f}%', (core, eff), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    # Gráfica 4: Comparación de métricas normalizadas
    # Normalizar todas las métricas para comparación visual
    norm_times = [t/max(times) for t in times]
    norm_speedups = [s/max(speedups) for s in speedups]
    norm_efficiency = [e/100 for e in efficiencies]  # Normalizar de 0-1
    
    x = np.arange(len(cores))
    width = 0.25
    
    ax4.bar(x - width, norm_times, width, label='Tiempo (normalizado)', alpha=0.8)
    ax4.bar(x, norm_speedups, width, label='Speedup (normalizado)', alpha=0.8)
    ax4.bar(x + width, norm_efficiency, width, label='Eficiencia (normalizada)', alpha=0.8)
    
    ax4.set_xlabel('Configuración de Cores')
    ax4.set_ylabel('Valor Normalizado (0-1)')
    ax4.set_title('Comparación de Métricas Normalizadas')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{c}c' for c in cores])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Guardar la gráfica
    graph_filename = 'rsa_performance_analysis.png'
    plt.savefig(graph_filename, dpi=300, bbox_inches='tight')
    print(f"\nGráficas guardadas en: {graph_filename}")
    
    # Mostrar la gráfica
    plt.show()
    
    return graph_filename

def create_vulnerability_summary_graph(vulnerable_pairs):
    """
    Crea una gráfica resumen de las vulnerabilidades encontradas
    """
    if not vulnerable_pairs:
        print("No hay pares vulnerables para graficar.")
        return
    
    # Extraer información de los pares vulnerables
    shared_primes = [pair[2] for pair in vulnerable_pairs]  # Los GCD (primos compartidos)
    
    # Análisis estadístico de los primos compartidos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Análisis de Vulnerabilidades RSA Encontradas', fontsize=16, fontweight='bold')
    
    # Gráfica 1: Distribución de tamaños de primos compartidos
    prime_lengths = [len(str(prime)) for prime in shared_primes]
    ax1.hist(prime_lengths, bins=20, alpha=0.7, color='red', edgecolor='black')
    ax1.set_xlabel('Número de Dígitos del Primo Compartido')
    ax1.set_ylabel('Frecuencia')
    ax1.set_title('Distribución del Tamaño de Primos Compartidos')
    ax1.grid(True, alpha=0.3)
    
    # Gráfica 2: Top 10 primos compartidos más frecuentes
    from collections import Counter
    prime_counts = Counter(shared_primes)
    most_common = prime_counts.most_common(min(10, len(prime_counts)))
    
    if most_common:
        primes, counts = zip(*most_common)
        # Usar solo los últimos dígitos para etiquetas si son muy largos
        labels = [str(p)[-8:] + '...' if len(str(p)) > 8 else str(p) for p in primes]
        
        ax2.bar(range(len(labels)), counts, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_xlabel('Primos Compartidos (últimos 8 dígitos)')
        ax2.set_ylabel('Frecuencia de Aparición')
        ax2.set_title('Primos Compartidos Más Frecuentes')
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Guardar la gráfica
    vuln_graph_filename = 'rsa_vulnerability_analysis.png'
    plt.savefig(vuln_graph_filename, dpi=300, bbox_inches='tight')
    print(f"Gráfica de vulnerabilidades guardada en: {vuln_graph_filename}")
    
    # Mostrar la gráfica
    plt.show()
    
    return vuln_graph_filename
    """
    Ejecuta el análisis completo con diferentes números de cores
    """
    print("=== RSA Shared Prime Attack Analysis ===")
    
    # Leer números del archivo
    print("Leyendo números del archivo integers.txt...")
    numbers = read_integers_file("integers.txt")
    
    if not numbers:
        print("No se pudieron cargar números del archivo.")
        return
    
    print(f"Números cargados: {len(numbers)}")
    print(f"Rango: {min(numbers)} - {max(numbers)}")
    
    # Configurar lista de cores a probar
    max_cores = cpu_count()
    core_counts = [1, 2, 4, 6, 8]
    # Agregar más valores hasta llegar al máximo disponible
    core_counts.extend(range(10, max_cores + 1, 2))
    # Asegurar que solo usamos cores disponibles
    core_counts = [c for c in core_counts if c <= max_cores]
    # Agregar el máximo disponible si no está
    if max_cores not in core_counts:
        core_counts.append(max_cores)
    
    print(f"Cores disponibles en el sistema: {max_cores}")
    print(f"Se probarán: {core_counts}")
    
    # Almacenar resultados para comparación
    timing_results = []
    all_vulnerable_pairs = None
    
    # Ejecutar análisis con diferentes números de cores
    for cores in core_counts:
        vulnerable_pairs, exec_time = analyze_with_cores(numbers, cores)
        timing_results.append((cores, exec_time, len(vulnerable_pairs)))
        
        # Guardar resultados de la primera ejecución para verificación
        if all_vulnerable_pairs is None:
            all_vulnerable_pairs = vulnerable_pairs
        
        # Verificar consistencia de resultados
        elif len(vulnerable_pairs) != len(all_vulnerable_pairs):
            print(f"ADVERTENCIA: Diferencia en resultados con {cores} cores!")
    
    # Guardar resultados vulnerables en archivo
    if all_vulnerable_pairs:
        output_filename = "vulnerable_rsa_pairs.txt"
        write_results_to_file(all_vulnerable_pairs, output_filename)
        print(f"\nResultados guardados en: {output_filename}")
        
        # Mostrar algunos ejemplos
        print("\nEjemplos de pares vulnerables encontrados:")
        for i, (a, b, gcd_val, factor_a, factor_b) in enumerate(all_vulnerable_pairs[:5]):
            print(f"  Par {i+1}:")
            print(f"    {a} = {factor_a} × {gcd_val}")
            print(f"    {b} = {factor_b} × {gcd_val}")
            print(f"    Factor compartido: {gcd_val}")
    
    # Mostrar análisis de rendimiento
    print(f"\n=== Análisis de Rendimiento ===")
    print("Cores\tTiempo(s)\tSpeedup\tEficiencia")
    print("-" * 45)
    
    baseline_time = timing_results[0][1]  # Tiempo con 1 core
    
    for cores, exec_time, vulnerable_count in timing_results:
        speedup = baseline_time / exec_time
        efficiency = speedup / cores * 100
        print(f"{cores}\t{exec_time:.2f}\t\t{speedup:.2f}x\t{efficiency:.1f}%")
    
    return timing_results, all_vulnerable_pairs

if __name__ == '__main__':
    # Verificar que existe el archivo de entrada
    if not os.path.exists("integers.txt"):
        print("Error: No se encuentra el archivo 'integers.txt'")
        print("Asegúrate de que el archivo esté en el directorio actual.")
        exit(1)
    
    # Ejecutar análisis completo
    timing_results, vulnerable_pairs = run_complete_analysis()
    
    print(f"\n=== Resumen Final ===")
    if vulnerable_pairs:
        print(f"Total de pares vulnerables: {len(vulnerable_pairs)}")
        print("Estos pares comparten factores primos, lo que los hace vulnerables")
        print("a ataques de factorización en el contexto de RSA.")
        
        # Crear gráfica de análisis de vulnerabilidades
        create_vulnerability_summary_graph(vulnerable_pairs)
    else:
        print("No se encontraron pares vulnerables en este conjunto de números.")
    
    # Crear gráficas de rendimiento
    print(f"\n=== Generando Gráficas de Rendimiento ===")
    create_performance_graphs(timing_results)
    
    print("\nAnálisis completado. ¡Revisa los resultados y las gráficas generadas!")