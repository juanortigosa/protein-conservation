# ConservationScope 🧬

App Streamlit para análisis de conservación de secuencias proteicas a partir de alineamientos múltiples.

## Funcionalidades

- **Tres métricas de scoring**: Jensen-Shannon Divergence, Shannon Entropy, Property Entropy
- **Gráfico interactivo** de scores a lo largo de la secuencia (Plotly)
- **Sequence Logo** SVG con colores por propiedad fisicoquímica
- **Tabla descargable** en CSV y Excel
- **Mapeo a estructura 3D**: genera scripts para PyMOL y ChimeraX

## Instalación local

```bash
# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Instalar dependencias
pip install -r requirements.txt

# Correr la app
streamlit run app.py
```

La app queda disponible en `http://localhost:8501`

## Deploy en Streamlit Community Cloud (gratis)

1. Subí la carpeta a un repositorio GitHub
2. Entrá a [share.streamlit.io](https://share.streamlit.io)
3. Conectá tu repo → seleccioná `app.py`
4. Click en **Deploy** — en ~2 minutos tenés una URL pública para compartir

## Formato de entrada

Alineamiento múltiple en formato **FASTA**:

```
>secuencia_1
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKAL---
>secuencia_2
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKAL---
...
```

- La **primera secuencia** se usa como referencia para la numeración
- Los gaps (`-`) son manejados automáticamente con penalización de gaps

## Métricas implementadas

| Métrica | Descripción |
|---|---|
| **Jensen-Shannon Divergence** | Divergencia entre distribución observada y fondo BLOSUM62. Valores altos = posición diverge del fondo = conservada/específica |
| **Shannon Entropy** | Entropía normalizada e invertida. Valores altos = baja diversidad = alta conservación |
| **Property Entropy** | Como Shannon pero sobre grupos fisicoquímicos (Taylor 1986). Detecta conservación funcional aunque cambien los residuos específicos |

## Pesos de secuencia

El pipeline implementa el esquema de **Henikoff & Henikoff (1994)**:
cada residuo en una columna contribuye `1/(k·r)` donde `k` = frecuencia del residuo
y `r` = número de tipos distintos. Reduce el sesgo cuando hay secuencias redundantes.

## Mapeo a estructura 3D

Los scripts generados asignan el score como **B-factor** de cada residuo.
Colorea la proteína: azul (baja conservación) → rojo (alta conservación).

**Requisito:** la numeración de residuos en el PDB debe coincidir con la columna `posicion`
(numeración de la secuencia de referencia en el MSA).
