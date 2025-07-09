import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import time

# Librerías para tuning
import optuna
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import ray

# ================================
# 1. CARGA Y PREPROCESAMIENTO DE DATOS
# ================================

def cargar_datos():
    """Carga y preprocesa el dataset de cáncer de mama"""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    print(f"Dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    print(f"Distribución de clases: {y.value_counts().to_dict()}")
    
    # Escalado de características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split 70/30
    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, stratify=y, random_state=42
    )
    
    print(f"Conjunto de entrenamiento: {x_train.shape[0]} muestras")
    print(f"Conjunto de prueba: {x_test.shape[0]} muestras")
    
    return x_train, x_test, y_train, y_test

# ================================
# 2. MODELO BASE (SIN TUNING)
# ================================

def modelo_base():
    """Evalúa el modelo base sin tuning"""
    print("\n" + "="*50)
    print("MODELO BASE (SIN TUNING)")
    print("="*50)
    
    x_train, x_test, y_train, y_test = cargar_datos()
    
    print(f"Dataset cargado: {x_train.shape[0]} muestras de entrenamiento, {x_test.shape[0]} de prueba")
    print(f"Características: {x_train.shape[1]}")
    
    # Modelo con parámetros por defecto
    modelo = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    start_time = time.time()
    modelo.fit(x_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = modelo.predict(x_test)
    f1 = f1_score(y_test, y_pred)
    
    print(f"F1 Score (modelo base): {f1:.4f}")
    print(f"Tiempo de entrenamiento: {training_time:.2f} segundos")
    
    return f1

# ================================
# 3. OPTIMIZACIÓN CON OPTUNA
# ================================

def objective_optuna(trial):
    """Función objetivo para Optuna"""
    x_train, x_test, y_train, y_test = cargar_datos()
    
    # Sugerencias de hiperparámetros
    n_estimators = trial.suggest_categorical('n_estimators', [50, 100, 150, 200])
    max_depth = trial.suggest_categorical('max_depth', [3, 5, 8, 10, None])
    min_samples_split = trial.suggest_categorical('min_samples_split', [2, 5, 10])
    
    # Modelo con parámetros sugeridos
    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1
    )
    
    # Validación cruzada con F1 score (reducir CV para consistencia)
    f1 = cross_val_score(modelo, x_train, y_train, cv=3, scoring="f1").mean()
    
    return f1

def optimizar_con_optuna():
    """Ejecuta la optimización con Optuna"""
    print("\n" + "="*50)
    print("OPTIMIZACIÓN CON OPTUNA")
    print("="*50)
    
    # Suprimir logs de Optuna para mayor claridad
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Crear estudio
    study = optuna.create_study(direction='maximize')
    
    start_time = time.time()
    study.optimize(objective_optuna, n_trials=15)  # Reducir para consistencia
    optuna_time = time.time() - start_time
    
    print(f"Tiempo total de optimización: {optuna_time:.2f} segundos")
    print(f"Mejor F1 Score (validación cruzada): {study.best_value:.4f}")
    print(f"Mejores hiperparámetros:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Evaluación final en conjunto de prueba
    x_train, x_test, y_train, y_test = cargar_datos()
    
    final_model = RandomForestClassifier(
        **study.best_params,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(x_train, y_train)
    y_pred = final_model.predict(x_test)
    final_f1 = f1_score(y_test, y_pred)
    
    print(f"F1 Score final en test: {final_f1:.4f}")
    
    return {
        'best_params': study.best_params,
        'best_cv_score': study.best_value,
        'final_f1': final_f1,
        'time': optuna_time,
        'n_trials': len(study.trials)
    }

# ================================
# 4. OPTIMIZACIÓN CON RAY TUNE
# ================================

def trainable_ray(config):
    """Función entrenable para Ray Tune"""
    try:
        x_train, x_test, y_train, y_test = cargar_datos()
        
        # Modelo con parámetros sugeridos por Ray Tune
        modelo = RandomForestClassifier(
            n_estimators=int(config["n_estimators"]),
            max_depth=config["max_depth"] if config["max_depth"] != "None" else None,
            min_samples_split=int(config["min_samples_split"]),
            random_state=42,
            n_jobs=1  # Cambiar a 1 para evitar conflictos con Ray
        )
        
        # Validación cruzada con F1 score
        f1_scores = cross_val_score(modelo, x_train, y_train, cv=3, scoring="f1")  # Reducir CV a 3
        f1 = f1_scores.mean()
        
        # Reportar resultado
        tune.report(f1_score=f1)
        
    except Exception as e:
        print(f"Error en trial: {e}")
        tune.report({"score": 0.0})  # Reportar score bajo en caso de error

def optimizar_con_ray_tune():
    """Ejecuta la optimización con Ray Tune"""
    print("\n" + "="*50)
    print("OPTIMIZACIÓN CON RAY TUNE")
    print("="*50)
    
    # Inicializar Ray con configuración más conservadora
    if ray.is_initialized():
        ray.shutdown()
    
    ray.init(
        ignore_reinit_error=True,
        num_cpus=2,  # Limitar CPUs para evitar sobrecarga
        include_dashboard=False,
        log_to_driver=False
    )
    
    try:
        # Espacio de búsqueda simplificado
        search_space = {
            "n_estimators": tune.choice([50, 100, 150, 200]),
            "max_depth": tune.choice([3, 5, 8, 10, None]),
            "min_samples_split": tune.choice([2, 5, 10]),
        }
        
        # Configuración más simple sin scheduler complejo
        start_time = time.time()
        
        # Ejecutar la búsqueda con configuración más robusta
        analysis = tune.run(
            trainable_ray,
        config=config,
        metric="score",  # debe coincidir con el nombre usado en tune.report
        mode="max",  # "max" si quieres maximizar, "min" si quieres minimizar
        num_samples=15
)

        ray_time = time.time() - start_time
        
        # Verificar que tenemos resultados
        if analysis.best_result is None:
            print("No se obtuvieron resultados válidos de Ray Tune")
            return None
            
        print(f"Tiempo total de optimización: {ray_time:.2f} segundos")
        print(f"Mejor F1 Score (validación cruzada): {analysis.best_result['f1_score']:.4f}")
        print("Mejores hiperparámetros:")
        for key, value in analysis.best_config.items():
            print(f"  {key}: {value}")
        
        # Evaluación final en conjunto de prueba
        x_train, x_test, y_train, y_test = cargar_datos()
        
        final_config = analysis.best_config.copy()
        if final_config['max_depth'] == 'None':
            final_config['max_depth'] = None
            
        final_model = RandomForestClassifier(
            n_estimators=int(final_config['n_estimators']),
            max_depth=final_config['max_depth'],
            min_samples_split=int(final_config['min_samples_split']),
            random_state=42,
            n_jobs=-1
        )
        final_model.fit(x_train, y_train)
        y_pred = final_model.predict(x_test)
        final_f1 = f1_score(y_test, y_pred)
        
        print(f"F1 Score final en test: {final_f1:.4f}")
        
        return {
            'best_params': analysis.best_config,
            'best_cv_score': analysis.best_result['f1_score'],
            'final_f1': final_f1,
            'time': ray_time,
            'n_trials': len([t for t in analysis.trials if t.status == 'TERMINATED'])
        }
        
    except Exception as e:
        print(f"Error en Ray Tune: {e}")
        return None
        
    finally:
        ray.shutdown()

# ================================
# 5. COMPARACIÓN DE RESULTADOS
# ================================

def comparar_resultados(modelo_base_f1, optuna_results, ray_results):
    """Compara los resultados de todas las estrategias"""
    print("\n" + "="*50)
    print("COMPARACIÓN DE RESULTADOS")
    print("="*50)
    
    if ray_results is None:
        print("Ray Tune no completó exitosamente. Comparando solo Optuna vs Modelo Base.")
        
        comparison_data = {
            'Método': ['Modelo Base', 'Optuna'],
            'F1 Score (Test)': [modelo_base_f1, optuna_results['final_f1']],
            'F1 Score (CV)': ['N/A', optuna_results['best_cv_score']],
            'Tiempo (s)': ['N/A', optuna_results['time']],
            'N° Trials': ['N/A', optuna_results['n_trials']]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        
        print(f"\nMEJORAS RESPECTO AL MODELO BASE:")
        print(f"Optuna: {((optuna_results['final_f1'] - modelo_base_f1) / modelo_base_f1 * 100):+.2f}%")
        
        return
    
    # Crear DataFrame para comparación
    comparison_data = {
        'Método': ['Modelo Base', 'Optuna', 'Ray Tune'],
        'F1 Score (Test)': [
            modelo_base_f1,
            optuna_results['final_f1'],
            ray_results['final_f1']
        ],
        'F1 Score (CV)': [
            'N/A',
            optuna_results['best_cv_score'],
            ray_results['best_cv_score']
        ],
        'Tiempo (s)': [
            'N/A',
            optuna_results['time'],
            ray_results['time']
        ],
        'N° Trials': [
            'N/A',
            optuna_results['n_trials'],
            ray_results['n_trials']
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    
    # Análisis de mejora
    print(f"\nMEJORAS RESPECTO AL MODELO BASE:")
    print(f"Optuna: {((optuna_results['final_f1'] - modelo_base_f1) / modelo_base_f1 * 100):+.2f}%")
    print(f"Ray Tune: {((ray_results['final_f1'] - modelo_base_f1) / modelo_base_f1 * 100):+.2f}%")
    
    # Reflexiones
    print(f"\nREFLEXIONES:")
    print(f"• Optuna completó {optuna_results['n_trials']} trials en {optuna_results['time']:.2f}s")
    print(f"• Ray Tune completó {ray_results['n_trials']} trials en {ray_results['time']:.2f}s")
    
    if optuna_results['time'] < ray_results['time']:
        print("• Optuna fue más rápido en la optimización")
    else:
        print("• Ray Tune fue más rápido en la optimización")
    
    if optuna_results['final_f1'] > ray_results['final_f1']:
        print("• Optuna encontró mejores hiperparámetros")
    else:
        print("• Ray Tune encontró mejores hiperparámetros")

# ================================
# 6. FUNCIÓN PRINCIPAL
# ================================

def main():
    """Función principal que ejecuta todo el análisis"""
    print("COMPARACIÓN DE ESTRATEGIAS DE TUNING AUTOMÁTICO")
    print("Dataset: Breast Cancer (Scikit-learn)")
    print("Modelo: Random Forest Classifier")
    print("Métrica: F1 Score")
    
    # 1. Modelo base
    base_f1 = modelo_base()
    
    # 2. Optimización con Optuna
    optuna_results = optimizar_con_optuna()
    
    # 3. Optimización con Ray Tune
    ray_results = optimizar_con_ray_tune()
    
    # 4. Comparación final
    comparar_resultados(base_f1, optuna_results, ray_results)
    
    print("\n" + "="*50)
    print("ANÁLISIS COMPLETADO")
    print("="*50)

if __name__ == "__main__":
    main()