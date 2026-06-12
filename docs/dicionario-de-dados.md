# Dicionário de Dados: Classificação Fitossanitária de Videiras

Este documento descreve a estrutura, o mapeamento e os atributos do conjunto de dados utilizado para o treinamento e avaliação do modelo de triagem primária (ResNet50 Suprema).

---

## 1. Estrutura Bruta de Arquivos (Imagens)
O *dataset* original é composto por imagens digitais coloridas (RGB) de folhas de videira, divididas fisicamente em diretórios por classe. 
* **Formato Original:** `.jpg` / `.png`
* **Canais de Cor:** 3 (RGB)
* **Resolução de Entrada da Rede:** Redimensionadas em tempo de execução para `224x224 pixels` (padrão exigido pela arquitetura ResNet50).

---

## 2. Metadados Tabulares
Para facilitar a ingestão de dados pelo TensorFlow e garantir a reprodutibilidade da separação dos conjuntos (*train_test_split* estratificado), a estrutura de pastas foi mapeada em dois arquivos de metadados:
1. `data/processed/metadados_treino.csv`
2. `data/processed/metadados_teste.csv`

### Estrutura das Colunas (CSVs)
| Nome da Coluna | Tipo de Dado | Descrição | Exemplo |
| :--- | :--- | :--- | :--- |
| `caminho_arquivo` | *String* | Caminho absoluto ou relativo apontando para a imagem física no disco. | `data/raw/Powdery Mildew/img_01.jpg` |
| `classe` | *String* | Rótulo textual da categoria fitossanitária à qual a folha pertence. | `Bacterial Leaf Spot` |
| `label` | *Integer* | Representação numérica da classe (variável *Target*), utilizada para o cálculo da função de perda (*Sparse Categorical Crossentropy*). | `2` |

---

## 3. Mapeamento da Variável Alvo (Target)
O problema consiste em uma classificação multiclasse (4 categorias). A conversão da variável categórica nominal (`classe`) para o formato numérico (`label`) obedece ao seguinte dicionário estrito:

* **`0` -> Healthy Leaves** (Folhas Saudáveis - Controle)
* **`1` -> Downy Mildew** (Míldio - Manchas cloróticas/oleosas globais)
* **`2` -> Bacterial Leaf Spot** (Mancha Bacteriana - Patologia minoritária, lesões necróticas puntiformes)
* **`3` -> Powdery Mildew** (Oídio - Revestimento pulverulento esbranquiçado)

---

## 4. Pré-processamento e Espaço Latente (Tensores)
Antes de os dados alimentarem a rede neural durante o treinamento, os atributos originais sofrem as seguintes transformações:

1. **Decodificação Estocástica (Treinamento):** Aplicação dinâmica de translações, zoom, rotações e espelhamentos para mitigação de Viés de Captura (*Shortcut Learning*).
2. **Normalização ResNet50:** Os tensores de imagem passam pela função nativa `tf.keras.applications.resnet50.preprocess_input`, que centraliza cada canal de cor em relação à média do dataset *ImageNet* (sem redimensionar a escala para 0-1, mantendo a ordem de grandeza de 0-255 e convertendo RGB para BGR).
3. **Vetor de Saída (Output):** O modelo retorna um vetor de dimensão `[1, 4]` contendo as probabilidades ativadas pela função *Softmax*. A classe predita é extraída via `argmax` do vetor.