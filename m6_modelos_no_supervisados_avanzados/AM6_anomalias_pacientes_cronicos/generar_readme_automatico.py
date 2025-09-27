# genera un readme.md a partir de las salidas del script principal
# uso: python generar_readme_automatico.py
import os
import pandas as pd

METRICAS = "metricas_clustering.csv"
CRUCE = "analisis_cruzado.csv"
ANOM_ISO = "anomalias_isolation_forest.csv"
ANOM_OCS = "anomalias_oneclass_svm.csv"

def leer_tabla(path):
    return pd.read_csv(path) if os.path.exists(path) else None

def main():
    tab = leer_tabla(METRICAS)
    cruce = leer_tabla(CRUCE)
    iso = leer_tabla(ANOM_ISO)
    ocs = leer_tabla(ANOM_OCS)

    n_iso = int(len(iso)) if iso is not None else 0
    n_ocs = int(len(ocs)) if ocs is not None else 0

    md = []
    md.append("# evaluación modular — módulo 6: segmentación y detección de anomalías (diabetes)\n")
    md.append("## 1. preprocesamiento y reducción de dimensionalidad\n")
    md.append("- escalado con standardscaler (variables numéricas).\n")
    md.append("- proyecciones 2d: pca, t-sne, umap.\n")

    md.append("\n## 2. segmentación (clustering)\n")
    if tab is not None and len(tab):
        md.append("\n**métricas de calidad**\n\n")
        md.append(tab.to_markdown(index=False))
        md.append("\n")
    else:
        md.append("- no se encontró metricas_clustering.csv\n")

    md.append("\n## 3. detección de anomalías\n")
    md.append(f"- isolation forest: **{n_iso}** casos\n")
    md.append(f"- one-class svm: **{n_ocs}** casos\n")

    md.append("\n## 4. análisis cruzado\n")
    if cruce is not None and len(cruce):
        md.append(cruce.to_markdown(index=True))
        md.append("\n")
    else:
        md.append("- no se encontró analisis_cruzado.csv\n")

    md.append("\n## 5. conclusiones y reflexión\n")
    md.append("- agregar interpretación final según resultados.\n")

    with open("readme.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("[ok] se generó readme.md")

if __name__ == "__main__":
    main()
