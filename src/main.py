import os
import random
import numpy as np
import tensorflow as tf
from pathlib import Path

from data.dataset import carregar_dados
from features.augmentation import obter_camada_augmentation
from models.resnet_otimizada import construir_modelo_otimizado
from evaluation.evaluate import avaliar_modelo, plotar_curvas_aprendizado
from visualization.visualize import gerar_auditoria_grad_cam

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def executar_pipeline():
    print("[INFO] Iniciando Pipeline Modular (ResNet50 Otimizada)...")
    
    DIRETORIO_RAIZ = Path(__file__).resolve().parent.parent
    PASTA_IMAGENS = DIRETORIO_RAIZ / "docs" / "imagens"
    
    MAPA_CLASSES = {"Healthy Leaves": 0, "Downy Mildew": 1, "Bacterial Leaf Spot": 2, "Powdery Mildew": 3}
    NOMES_CLASSES = list(MAPA_CLASSES.keys())
    
    ds_treino, ds_val, ds_teste, df_teste = carregar_dados(
        caminho_treino=str(DIRETORIO_RAIZ / "data/processed/metadados_treino.csv"),
        caminho_teste=str(DIRETORIO_RAIZ / "data/processed/metadados_teste.csv"),
        mapa_classes=MAPA_CLASSES, seed=SEED
    )
    
    pasta_modelos = DIRETORIO_RAIZ / "models"
    pasta_modelos.mkdir(parents=True, exist_ok=True)
    caminho_modelo = pasta_modelos / "modelo_campeao_resnet50_otimizada.keras"
    
    historico = None
    
    if caminho_modelo.exists():
        print(f"\n[INFO] Modelo pré-treinado detectado em: {caminho_modelo}")
        print("[INFO] Omitindo fase de treinamento. Carregando pesos na memória...")
        modelo = tf.keras.models.load_model(caminho_modelo)
    else:
        print("\n[INFO] Modelo não encontrado. Iniciando Treinamento do zero...")
        camada_aug = obter_camada_augmentation(SEED)
        modelo = construir_modelo_otimizado(camada_aug, SEED)
        
        pesos_de_classe = {0: 0.54, 1: 0.71, 2: 6.81, 3: 1.68}
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)]
        
        historico = modelo.fit(ds_treino, validation_data=ds_val, epochs=15, class_weight=pesos_de_classe, callbacks=callbacks)
        
        modelo.save(caminho_modelo)
        print(f"\n[SUCESSO] Modelo salvo em {caminho_modelo}")
        
    if historico:
        plotar_curvas_aprendizado(historico, PASTA_IMAGENS)
    else:
        print("\n[INFO] Gráficos de aprendizado omitidos (modelo pré-treinado não possui histórico de época ativo).")
        
    avaliar_modelo(modelo, ds_teste, NOMES_CLASSES, PASTA_IMAGENS)
    gerar_auditoria_grad_cam(modelo, df_teste, NOMES_CLASSES, PASTA_IMAGENS, DIRETORIO_RAIZ)
    
    print("\n[INFO] Execução do Pipeline Finalizada com Sucesso! =)")

if __name__ == "__main__":
    executar_pipeline()