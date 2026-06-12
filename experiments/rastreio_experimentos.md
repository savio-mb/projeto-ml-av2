# Rastreio de Experimentos e Hiperparâmetros

Este documento atende aos requisitos **RQ-PM-01** e **RQ-PM-02**, registrando o rastreio de modelos, hiperparâmetros e métricas de desempenho.

---

## 1. Fase de Triagem (Screening Inicial)

**Objetivo:** Avaliar o baseline de três arquiteturas distintas via Transfer Learning, sem aplicação de balanceamento avançado.

**Hiperparâmetros Globais (Aplicados a todos os testes):**
* **Otimizador:** Adam (Taxa de aprendizagem inicial = 1e-3)
* **Função de Perda:** Sparse Categorical Crossentropy
* **Épocas:** 5
* **Batch Size:** 32
* **Top Layers:** Pesos base congelados, Global Average Pooling, Camada Densa com função Softmax.

### Tabela de Comparação de Métricas
| Modelo | Acurácia (Treino) | Acurácia (Teste) | Recall (Mancha Bact.) | Decisão |
| :--- | :--- | :--- | :--- | :--- |
| **Xception** | 92.10% | 90.65% | 50.0% | Descartado (Alto viés) |
| **MobileNetV2** | 95.50% | 94.13% | 60.0% | Descartado (Baixa sensibilidade) |
| **ResNet50** | **97.80%** | **96.88%** | **85.0%** | **Aprovado (McNemar p < 0.05)** |

---

## 2. Fase de Otimização e Validação (Modelo Definitivo)

**Objetivo:** Otimizar a arquitetura campeã (ResNet50) aplicando regularização, mitigando o desbalanceamento e atestando a estabilidade de generalização.

### Tabela de Hiperparâmetros Ajustados e Resultados
| Parâmetro / Métrica | Valor Ajustado / Resultado Final |
| :--- | :--- |
| **Modelo Base** | ResNet50 (Transfer Learning) |
| **Épocas (Epochs)** | 10 (com Early Stopping - patience: 3) |
| **Batch Size** | 32 |
| **Class Weights** | Saudável: 0.54 \| Míldio: 0.71 \| Mancha Bact.: 6.81 \| Oídio: 1.68 |
| **Data Augmentation** | Ativado (Rotação, Espelhamento, Zoom negativo, Translação) |
| **Acurácia CV (5-Fold)** | **88.26%** |
| **Desvio Padrão CV** | **± 1.89%** |

### Observações Finais
A ResNet50 Otimizada foi aprovada como versão definitiva. A injeção de pesos assimétricos severos na função de perda (6.81x) e as mutações espaciais contiveram com sucesso o viés de captura (*Shortcut Learning*) contra a classe minoritária. A altíssima estabilidade atestada pela Validação Cruzada K-Fold comprova a robustez do modelo face à variância do *dataset*.