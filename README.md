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
git clone https://github.com/savio-mb/projeto-ml-av2.git
cd projeto-ml-av2