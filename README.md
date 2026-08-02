
# Actividad 4: Comparación de modelos preentrenados de Hugging Face

## Descripción del proyecto

Este repositorio contiene el prototipo desarrollado para la Actividad 4 de la materia Gestión de Proyectos de Inteligencia Artificial. El objetivo del proyecto es implementar, evaluar y comparar modelos preentrenados de Hugging Face para un caso de uso de Procesamiento de Lenguaje Natural.

## Rama de inteligencia artificial seleccionada

Procesamiento de Lenguaje Natural, NLP.

## Caso de uso

Clasificación automática de sentimiento en textos. El prototipo clasifica frases como positivas o negativas utilizando modelos preentrenados disponibles en Hugging Face.

## Dataset

Se utilizó el dataset GLUE SST-2, cargado mediante la librería datasets de Hugging Face.

Procedimiento de carga:

load_dataset("nyu-mll/glue", "sst2")

Campos principales:

- text: texto o frase a clasificar.
- true_label: etiqueta numérica original.
- true_label_text: etiqueta textual.
- idx: identificador original del registro.

## Modelos evaluados

1. distilbert/distilbert-base-uncased-finetuned-sst-2-english
2. textattack/bert-base-uncased-SST-2
3. siebert/sentiment-roberta-large-english

## Métricas utilizadas

- Accuracy
- Precision
- Recall
- F1-score
- Latencia promedio por inferencia

## Entorno de ejecución

El prototipo fue desarrollado en Google Colab utilizando Python y, cuando estuvo disponible, aceleración por GPU.

## Dependencias principales

- transformers
- datasets
- evaluate
- accelerate
- scikit-learn
- pandas
- numpy
- openpyxl
- huggingface_hub
- torch

## Resultados

Los resultados comparativos se encuentran en la carpeta results.

Archivos principales:

- results/model_comparison_results.csv
- results/model_comparison_summary.xlsx
- results/detailed_model_predictions.csv
- results/detailed_model_predictions.xlsx

## Modelo recomendado

El modelo recomendado de acuerdo con los resultados obtenidos fue:

**RoBERTa Large Sentiment**

Este modelo obtuvo:

- Accuracy: 0.9450
- Precision: 0.9375
- Recall: 0.9633
- F1-score: 0.9502
- Latencia promedio: 0.0165 segundos

## Conclusión

La evaluación permitió identificar que la selección de un modelo preentrenado debe considerar tanto el desempeño predictivo como la eficiencia computacional. El F1-score permitió comparar el balance entre precision y recall, mientras que la latencia permitió analizar la viabilidad del modelo para una posible implementación práctica.
