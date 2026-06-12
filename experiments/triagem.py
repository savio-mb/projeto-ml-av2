import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from pathlib import Path

# Configuração de rotas
DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_RAIZ = DIRETORIO_ATUAL.parent
DIRETORIO_SRC = DIRETORIO_RAIZ / "src"
sys.path.append(str(DIRETORIO_SRC))

from data.dataset import carregar_dados

def executar_triagem():
    arquivo_log = DIRETORIO_ATUAL / "log_triagem_oficial.csv"
    
    # ==============================================================================
    # 1. VERIFICAÇÃO DE CACHE (LOG)
    # ==============================================================================
    if arquivo_log.exists():
        print(f"[INFO] Log de experimento encontrado em {arquivo_log}.")
        print("[INFO] Pulando treinamento pesado e carregando dados históricos...")
        df_resultados = pd.read_csv(arquivo_log)
    
    else:
        print("[INFO] Nenhum log encontrado. Iniciando Experimento de Triagem do zero (5 épocas/modelo)...")
        
        SEED = 42
        MAPA_CLASSES = {"Healthy Leaves": 0, "Downy Mildew": 1, "Bacterial Leaf Spot": 2, "Powdery Mildew": 3}
        NOMES_CLASSES = list(MAPA_CLASSES.keys())
        
        caminho_treino = str(DIRETORIO_RAIZ / "data/processed/metadados_treino.csv")
        caminho_teste = str(DIRETORIO_RAIZ / "data/processed/metadados_teste.csv")
        
        ds_treino, ds_val, ds_teste, df_teste = carregar_dados(
            caminho_treino=caminho_treino, caminho_teste=caminho_teste, mapa_classes=MAPA_CLASSES, seed=SEED, batch_size=32
        )
        
        y_true = np.concatenate([y for x, y in ds_teste], axis=0)
        resultados_triagem = {'Arquitetura': [], 'Acurácia Global (%)': [], 'Recall (Mancha Bact.) (%)': []}

        modelos_candidatos = {
            'ResNet50': (tf.keras.applications.ResNet50, tf.keras.applications.resnet50.preprocess_input),
            'MobileNetV2': (tf.keras.applications.MobileNetV2, tf.keras.applications.mobilenet_v2.preprocess_input),
            'Xception': (tf.keras.applications.Xception, tf.keras.applications.xception.preprocess_input)
        }

        for nome, (arquitetura, preprocess_func) in modelos_candidatos.items():
            print(f"\n[Treinando] {nome} Baseline...")
            inputs = tf.keras.Input(shape=(224, 224, 3))
            x = preprocess_func(inputs)
            base_model = arquitetura(weights='imagenet', include_top=False, input_tensor=x)
            base_model.trainable = False 
            
            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
            
            modelo_triagem = tf.keras.Model(inputs, outputs)
            modelo_triagem.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            modelo_triagem.fit(ds_treino, validation_data=ds_val, epochs=5, verbose=1)
            
            y_pred = np.argmax(modelo_triagem.predict(ds_teste, verbose=0), axis=1)
            report = classification_report(y_true, y_pred, target_names=NOMES_CLASSES, output_dict=True, zero_division=0)
            
            resultados_triagem['Arquitetura'].append(nome)
            resultados_triagem['Acurácia Global (%)'].append(report['accuracy'] * 100)
            resultados_triagem['Recall (Mancha Bact.) (%)'].append(report['Bacterial Leaf Spot']['recall'] * 100)

        # Salva o resultado para execuções futuras
        df_resultados = pd.DataFrame(resultados_triagem)
        df_resultados.to_csv(arquivo_log, index=False)
        print(f"\n[SUCESSO] Treinamento concluído. Log salvo em {arquivo_log}")

    # ==============================================================================
    # 2. GERAÇÃO DO GRÁFICO (DESIGN CORRIGIDO)
    # ==============================================================================
    print("\n[INFO] Gerando gráfico comparativo...")
    df_comparativo = df_resultados.melt(id_vars='Arquitetura', var_name='Métrica', value_name='Valor (%)')
    
    plt.figure(figsize=(11, 6)) 
    ax = sns.barplot(data=df_comparativo, x='Arquitetura', y='Valor (%)', hue='Métrica', palette=['#2c3e50', '#e74c3c'])
    
    plt.title('Comparativo de Desempenho Arquitetural (Screening)', fontweight='bold', pad=20)
    plt.ylim(0, 115) 
    plt.xlabel('Arquitetura', fontweight='bold', labelpad=15)
    plt.ylabel('Valor (%)', fontweight='bold', labelpad=10)
    plt.legend(title='Métrica', bbox_to_anchor=(1.02, 1), loc='upper left')

    for p in ax.patches:
        altura = p.get_height()
        if altura > 0:
            ax.annotate(f'{altura:.1f}%', (p.get_x() + p.get_width() / 2., altura), 
                        ha='center', va='bottom', fontweight='bold', xytext=(0, 5), textcoords='offset points')

    plt.tight_layout()
    
    pasta_imagens = DIRETORIO_RAIZ / "docs" / "imagens"
    pasta_imagens.mkdir(parents=True, exist_ok=True)
    caminho_salvar = pasta_imagens / "comparativo_modelos_real.png"
    
    plt.savefig(caminho_salvar, dpi=300, bbox_inches='tight')
    print(f"[SUCESSO] Gráfico salvo em: {caminho_salvar}")

if __name__ == "__main__":
    executar_triagem()