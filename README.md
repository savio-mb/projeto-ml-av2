# Classificação Fitossanitária de Folhas de Videira utilizando Deep Learning

## 📌 Visão Executiva do Projeto
Este repositório contém o ciclo completo (ponta a ponta) de um projeto de Machine Learning focado em Visão Computacional. O objetivo principal é diagnosticar patologias em folhas de videira a partir de imagens, fornecendo uma ferramenta preditiva viável para o manejo agrícola.

* **Problema:** Identificação tardia de doenças fúngicas e bacterianas em plantações, resultando em perdas de safra.
* **Objetivo:** Desenvolver e validar um modelo de Deep Learning capaz de classificar patologias com alta sensibilidade para doenças raras.
* **Variável-Alvo:** Classificação multiclasse categórica em 4 rótulos: *Healthy Leaves* (Saudável), *Downy Mildew* (Míldio), *Bacterial Leaf Spot* (Mancha Bacteriana) e *Powdery Mildew* (Oídio).

## 📊 Origem e Resumo dos Dados
Os dados são originários do *dataset* NGLD. O conjunto apresenta um grave desbalanceamento natural estrutural, possuindo:
* Ampla predominância de amostras de folhas saudáveis (1.254 amostras).
* Uma classe minoritária crítica (Mancha Bacteriana), exigindo abordagens de regularização matemática (*Class Weights*) e mutação espacial estocástica (*Data Augmentation*) para evitar o colapso preditivo.

*Nota de Versionamento:* Em conformidade com as boas práticas e segurança de dados, os arquivos brutos de imagem não são versionados neste repositório.

## 🚀 Instruções de Instalação e Execução
A execução deste projeto exige um ambiente limpo com suporte a processamento de tensores.

1. **Clone o repositório:**
```bash
git clone [https://github.com/savio-mb/projeto-ml-av2.git](https://github.com/savio-mb/projeto-ml-av2.git)
cd projeto-ml-av2
```

2. **Baixe o Dataset:**
Faça o download das imagens originais diretamente do repositório oficial da Mendeley Data:
🔗 [NGLD Dataset - Mendeley Data](https://data.mendeley.com/datasets/8nnd2ypcv3/5)
Extraia o conteúdo do arquivo baixado e coloque as pastas das classes diretamente no diretório `data/raw/` do seu projeto local.

3. **Instale as dependências rigorosas:**
```bash
pip install -r requirements.txt
```

4. **(Opcional) Download dos Pesos Pré-treinados:**
Para evitar o tempo de treinamento local (que pode ser longo em máquinas sem GPU), você pode baixar os pesos do modelo definitivo já treinado.
🔗 [Download do Modelo ResNet50 (Google Drive) - 93MB](https://drive.google.com/drive/folders/1n7QXYXxtktLGqHYgwXG14p_VI8RFwBHv?usp=sharing)
*Coloque o arquivo baixado diretamente na pasta `models/`.*

5. **Comando único de execução do pipeline:**
Para executar o carregamento de dados, treinar o modelo definitivo, extrair métricas e rodar a auditoria visual (Grad-CAM) sequencialmente, utilize o comando orquestrador:
```bash
python src/main.py
```

### ⚠️ Nota sobre Aceleração de Hardware (GPU) no Windows
O arquivo `requirements.txt` instalará a versão mais recente e estável do TensorFlow. Por padrão, a partir da versão 2.11, o TensorFlow **não oferece suporte nativo para aceleração por GPU (NVIDIA) no ambiente Windows clássico**, executando o treinamento através da CPU. 

Se você deseja executar este projeto utilizando a placa de vídeo no Windows para acelerar o treinamento, você tem duas opções:
1. **(Recomendado)** Executar o projeto através do **WSL2** (Windows Subsystem for Linux), onde o suporte à GPU para as versões mais recentes do TensorFlow é totalmente nativo.
2. Fazer o downgrade forçado para a última versão com suporte nativo no Windows clássico, alterando no arquivo de dependências para `tensorflow==2.10.0` (exige CUDA Toolkit 11.2 e cuDNN 8.1 instalados na máquina).

## 🏆 Resumo dos Resultados e Limitações
* **Resultados e Desempenho:** A arquitetura selecionada, ResNet50 otimizada, atingiu **90,0% de Acurácia Global** no particionamento de teste. O modelo mitigou com sucesso os Falsos Negativos da classe minoritária crítica, atingindo a marca de **100% de Recall** na identificação da Mancha Bacteriana.
* **Validação Estatística:** O Teste Estatístico de McNemar (p-value < 0.05) provou matematicamente que a superioridade da ResNet50 sobre arquiteturas concorrentes testadas na triagem inicial é estatisticamente significativa e não decorrente de ruído amostral.
* **Limitações e Ameaças à Validade:** A auditoria de interpretabilidade visual (Grad-CAM) revelou que, devido às condições fotográficas de laboratório do *dataset*, o modelo atingiu sua alta sensibilidade valendo-se parcialmente de *Shortcut Learning* (ancoragem nas sombras e no contraste do fundo em vez da anatomia microscópica). Adicionalmente, identificou-se sobreposição visual dificultando a separação entre amostras de Míldio e folhas saudáveis. Para aplicações em campo aberto, é essencial adicionar uma etapa prévia de segmentação (remoção de fundo) por software para garantir total interpretabilidade biológica.