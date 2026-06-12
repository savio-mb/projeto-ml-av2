import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

# ==============================================================================
# CONFIGURAÇÃO DE ROTAS
# ==============================================================================
DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_RAIZ = DIRETORIO_ATUAL.parent
sys.path.append(str(DIRETORIO_RAIZ / "src"))

from features.augmentation import obter_camada_augmentation
from models.resnet_otimizada import construir_modelo_otimizado

def executar_kfold():
    print("[INFO] Iniciando Validação Cruzada Estratificada (5-Fold)...")
    print("[INFO] Aceleradores de GPU detectados. Preparando tensores...")
    
    # 1. Carregando e unindo o metadado inteiro para o K-Fold poder fatiar
    df_treino = pd.read_csv(DIRETORIO_RAIZ / "data/processed/metadados_treino.csv")
    df_teste = pd.read_csv(DIRETORIO_RAIZ / "data/processed/metadados_teste.csv")
    df_total = pd.concat([df_treino, df_teste], ignore_index=True)

    # 2. Mapeamento à prova de falhas (igual ao da EDA)
    def mapear_caminho(caminho):
        caminho_limpo = str(caminho).replace('\\', '/')
        if "data/raw/" in caminho_limpo:
            sufixo = caminho_limpo.split("data/raw/")[-1]
            return str(DIRETORIO_RAIZ / "data" / "raw" / sufixo)
        return str(DIRETORIO_RAIZ / caminho_limpo)

    caminhos = df_total['caminho_arquivo'].apply(mapear_caminho).values
    mapa_classes = {"Healthy Leaves": 0, "Downy Mildew": 1, "Bacterial Leaf Spot": 2, "Powdery Mildew": 3}
    labels = df_total['classe'].map(mapa_classes).values

    # 3. Função de leitura dinâmica para a GPU
    def processar_caminho(caminho, label):
        img = tf.io.read_file(caminho)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        return tf.keras.applications.resnet50.preprocess_input(img), label

    # 4. Configuração do 5-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acuracias = []
    pesos_de_classe = {0: 0.54, 1: 0.71, 2: 6.81, 3: 1.68}

    # 5. Loop de Treinamento
    for fold, (idx_treino, idx_val) in enumerate(skf.split(caminhos, labels), 1):
        print(f"\n==================================================")
        print(f"            PROCESSANDO FOLD {fold}/5")
        print(f"==================================================")
        
        x_treino, y_treino = caminhos[idx_treino], labels[idx_treino]
        x_val, y_val = caminhos[idx_val], labels[idx_val]

        ds_treino = tf.data.Dataset.from_tensor_slices((x_treino, y_treino)) \
            .map(processar_caminho, num_parallel_calls=tf.data.AUTOTUNE) \
            .batch(32).prefetch(tf.data.AUTOTUNE)
            
        ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)) \
            .map(processar_caminho, num_parallel_calls=tf.data.AUTOTUNE) \
            .batch(32).prefetch(tf.data.AUTOTUNE)

        # Destrói o modelo anterior da memória da placa de vídeo para não vazar dados
        tf.keras.backend.clear_session()
        
        camada_aug = obter_camada_augmentation(42)
        modelo = construir_modelo_otimizado(camada_aug, 42)

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)]
        
        modelo.fit(ds_treino, validation_data=ds_val, epochs=10, class_weight=pesos_de_classe, callbacks=callbacks, verbose=1)
        
        # Avaliação final do Fold
        _, acc = modelo.evaluate(ds_val, verbose=0)
        print(f"\n[RESULTADO] Acurácia do Fold {fold}: {acc * 100:.2f}%")
        acuracias.append(acc * 100)

    # 6. Cálculo Estatístico
    media = np.mean(acuracias)
    desvio = np.std(acuracias)

    print("\n==================================================")
    print("      RESULTADO DA VALIDAÇÃO CRUZADA (5-FOLD)")
    print("==================================================")
    print(f"Acurácia Média   : {media:.2f}%")
    print(f"Desvio Padrão    : ± {desvio:.2f}%")
    print("==================================================")

    # 7. Salvando Log para o repositório
    caminho_log = DIRETORIO_ATUAL / "log_kfold.txt"
    with open(caminho_log, "w", encoding="utf-8") as f:
        f.write("RELATÓRIO DE VALIDAÇÃO CRUZADA (5-FOLD)\n")
        f.write("Modelo: ResNet50 Otimizada\n")
        f.write("-" * 45 + "\n")
        for i, acc in enumerate(acuracias, 1):
            f.write(f"Fold {i}: {acc:.2f}%\n")
        f.write("-" * 45 + "\n")
        f.write(f"Média Final    : {media:.2f}%\n")
        f.write(f"Desvio Padrão  : ± {desvio:.2f}%\n")

    print(f"\n[SUCESSO] Prova estatística de K-Fold salva em {caminho_log}")

if __name__ == "__main__":
    executar_kfold()