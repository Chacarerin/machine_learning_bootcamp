# 🔎 Módulo 9: Interpretabilidad y Explicabilidad

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

En la actualidad, a medida que los ensambles algorítmicos y las redes profundas se vuelven más ubicuas, su opacidad estructural (comportamiento de "Caja Negra") supone un riesgo analítico. Este módulo enfoca sus estudios en la IA Explicable (*Explainable AI - XAI*), proporcionando marcos teóricos para auditar modelos empíricamente.

## 📌 Contenidos Principales
- **Ética y Gobernanza Algorítmica:** Análisis del impacto legal, el sesgo algorítmico y la necesidad humana de confiar en los dictámenes automatizados en áreas críticas (medicina, finanzas).
- **Importancia Global de Variables:** Métodos tradicionales basados en modelos interpretables (como la profundidad de los árboles de decisión) y evaluación intrínseca de permutaciones (*Permutation Importance*).
- **Interpretación Agnóstica Local:**
  - **LIME:** Generación de modelos lineales sucedáneos alrededor de una predicción individual, aislando qué variables condujeron a esa decisión específica.
  - **Valores SHAP:** Marco analítico basado fundamentalmente en la teoría de juegos cooperativos, garantizando una asignación justa y equitativa de las contribuciones marginales de cada característica.

## ⚙️ Tecnologías y Frameworks Aplicados
- Integración de paquetes open-source especializados (`shap`, `lime`) aplicados retroactivamente sobre estimadores entrenados en Scikit-Learn y Keras, transformando predicciones opacas en gráficas de fuerzas (Force Plots) analizables por un ojo humano.

---
*Desarrollado por Rubén Schnettler.*
