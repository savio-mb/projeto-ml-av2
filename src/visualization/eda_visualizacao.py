import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_RAIZ = DIRETORIO_ATUAL.parent.parent
PASTA_IMAGENS = DIRETORIO_RAIZ / "docs" / "imagens"
PASTA_IMAGENS.mkdir(parents=True, exist_ok=True)

def executar_eda():
    print("[INFO] Iniciando Análise Exploratória de Dados (EDA)...")
    
    caminho_treino = DIRETORIO_RAIZ / "data/processed/metadados_treino.csv"
    caminho_teste = DIRETORIO_RAIZ / "data/processed/metadados_teste.csv"
    
    df_treino = pd.read_csv(caminho_treino)
    df_teste = pd.read_csv(caminho_teste)
    df_total = pd.concat([df_treino, df_teste], ignore_index=True)
    
    def mapear_caminho_real(caminho_csv):
        
        caminho_limpo = str(caminho_csv).replace('\\', '/')
        if "data/raw/" in caminho_limpo:
            sufixo = caminho_limpo.split("data/raw/")[-1]
            caminho_final = DIRETORIO_RAIZ / "data" / "raw" / sufixo
        else:
            caminho_final = DIRETORIO_RAIZ / caminho_limpo
        return str(caminho_final)
        
    df_total['caminho_absoluto'] = df_total['caminho_arquivo'].apply(mapear_caminho_real)
    
    
    caminho_teste_radar = df_total['caminho_absoluto'].iloc[0]
    if not os.path.exists(caminho_teste_radar):
        print(f"\n[ALERTA CRÍTICO] O OpenCV não está encontrando as imagens físicas!")
        print(f"O script tentou abrir esta imagem e falhou:\n -> {caminho_teste_radar}")
        print("Verifique se as pastas dentro de data/raw/ estão realmente neste local.")
        return 
    else:
        print(f"[OK] Imagens físicas localizadas com sucesso! Montando gráficos...\n")

    sns.set_theme(style="whitegrid")

   
    print("[1/5] Gerando Distribuição de Classes...")
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(data=df_total, y='classe', order=df_total['classe'].value_counts().index, palette='viridis')
    plt.title('Distribuição de Imagens por Classe Fitossanitária', fontweight='bold', pad=15)
    plt.xlabel('Número de Imagens')
    plt.ylabel('Classe')
    for p in ax.patches:
        ax.annotate(f'{int(p.get_width())}', (p.get_width() + 10, p.get_y() + 0.5), ha='left', va='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(PASTA_IMAGENS / "eda_01_distribuicao_classes.png", dpi=300)
    plt.close()

    
    print("[2/5] Gerando Mosaico de Amostras...")
    classes = df_total['classe'].unique()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for i, cls in enumerate(classes):
        amostras = df_total[df_total['classe'] == cls]['caminho_absoluto'].values
        for amostra_path in amostras:
            
            img = cv2.imread(amostra_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i].imshow(img_rgb)
                axes[i].set_title(cls, fontweight='bold')
                axes[i].axis('off')
                break 
               
    plt.tight_layout()
    plt.savefig(PASTA_IMAGENS / "eda_02_mosaico_amostras.png", dpi=300)
    plt.close()

    
    print("[3/5] Gerando Proporção de Particionamento...")
    tamanhos = [len(df_treino), len(df_teste)]
    labels = [f'Treino\n({tamanhos[0]})', f'Teste\n({tamanhos[1]})']
    plt.figure(figsize=(6, 6))
    plt.pie(tamanhos, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#3498db', '#e74c3c'], 
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}, textprops={'fontweight': 'bold', 'fontsize': 12})
    plt.title('Divisão do Dataset (Hold-out)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(PASTA_IMAGENS / "eda_03_proporcao_treino_teste.png", dpi=300)
    plt.close()

   
    print("[4/5] Calculando Intensidade de Pixels (Isso demora alguns segundos)...")
    df_amostra = df_total.groupby('classe').head(100)
    
    intensidades = {'classe': [], 'brilho_medio': [], 'contraste_medio': []}
    
    for _, row in df_amostra.iterrows():
        img = cv2.imread(row['caminho_absoluto'], cv2.IMREAD_GRAYSCALE)
        if img is not None:
            intensidades['classe'].append(row['classe'])
            intensidades['brilho_medio'].append(np.mean(img))
            intensidades['contraste_medio'].append(np.std(img))

   
    if len(intensidades['classe']) == 0:
        print("[ERRO] Nenhuma imagem pôde ser lida pelo OpenCV para os gráficos 4 e 5.")
        return

    df_intensidades = pd.DataFrame(intensidades)

    print("[5/5] Gerando Gráficos de Brilho e Contraste...")
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_intensidades, x='brilho_medio', y='classe', palette='Set2')
    plt.title('Distribuição de Brilho Médio por Classe', fontweight='bold')
    plt.xlabel('Intensidade Média de Pixel (0-255)')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(PASTA_IMAGENS / "eda_04_brilho_medio.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_intensidades, x='contraste_medio', y='classe', palette='Set3')
    plt.title('Distribuição de Contraste (Desvio Padrão) por Classe', fontweight='bold')
    plt.xlabel('Desvio Padrão de Pixel')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(PASTA_IMAGENS / "eda_05_contraste_medio.png", dpi=300)
    plt.close()

    print("[SUCESSO] Todos os 5 gráficos da EDA foram salvos em docs/imagens/")

if __name__ == "__main__":
    executar_eda()