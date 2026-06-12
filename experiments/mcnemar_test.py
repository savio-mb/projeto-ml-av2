import sys
import numpy as np
import tensorflow as tf
from statsmodels.stats.contingency_tables import mcnemar
from pathlib import Path

# ==============================================================================
# CONFIGURAÇÃO DE ROTAS (Para acessar a pasta src)
# ==============================================================================
DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_RAIZ = DIRETORIO_ATUAL.parent
DIRETORIO_SRC = DIRETORIO_RAIZ / "src"
sys.path.append(str(DIRETORIO_SRC))

from data.dataset import carregar_dados

def executar_teste_mcnemar():
    print("[INFO] Iniciando Teste Estatístico de McNemar...")
    print("[INFO] Treinando ResNet50 e MobileNetV2 (5 épocas) para extrair predições...")
    print("[AVISO] Isso pode levar alguns minutos na GPU...\n")

    SEED = 42
    MAPA_CLASSES = {"Healthy Leaves": 0, "Downy Mildew": 1, "Bacterial Leaf Spot": 2, "Powdery Mildew": 3}

    # Carregando o pipeline de dados
    ds_treino, ds_val, ds_teste, df_teste = carregar_dados(
        caminho_treino=str(DIRETORIO_RAIZ / "data/processed/metadados_treino.csv"),
        caminho_teste=str(DIRETORIO_RAIZ / "data/processed/metadados_teste.csv"),
        mapa_classes=MAPA_CLASSES, seed=SEED, batch_size=32
    )

    y_true = np.concatenate([y for x, y in ds_teste], axis=0)

    # Função auxiliar para treinar rapidamente e cuspir o array de predições
    def treinar_extrair_predicoes(arquitetura, preprocess_func, nome):
        print(f"\n--- Treinando {nome} (Baseline) ---")
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = preprocess_func(inputs)
        
        base_model = arquitetura(weights='imagenet', include_top=False, input_tensor=x)
        base_model.trainable = False 
        
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
        
        modelo = tf.keras.Model(inputs, outputs)
        modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Treinamento enxuto apenas para coletar os vetores de decisão
        modelo.fit(ds_treino, validation_data=ds_val, epochs=5, verbose=1)
        
        predicoes = np.argmax(modelo.predict(ds_teste, verbose=0), axis=1)
        return predicoes

    # 1. Extração dos vetores de predição
    y_pred_resnet = treinar_extrair_predicoes(tf.keras.applications.ResNet50, tf.keras.applications.resnet50.preprocess_input, "ResNet50")
    y_pred_mobile = treinar_extrair_predicoes(tf.keras.applications.MobileNetV2, tf.keras.applications.mobilenet_v2.preprocess_input, "MobileNetV2")

    print("\n[INFO] Calculando Tabela de Contingência...")
    
    # 2. Construindo as variáveis do McNemar
    acertos_resnet = (y_pred_resnet == y_true)
    acertos_mobile = (y_pred_mobile == y_true)

    # b: ResNet acertou e MobileNet errou
    b = np.sum(acertos_resnet & ~acertos_mobile)
    
    # c: ResNet errou e MobileNet acertou
    c = np.sum(~acertos_resnet & acertos_mobile)

    tabela_contingencia = [[0, b], 
                           [c, 0]]

    # 3. Executando o teste estatístico (usando aproximação Qui-Quadrado com correção de continuidade)
    resultado = mcnemar(tabela_contingencia, exact=False, correction=True)

    print("\n==================================================")
    print("      RESULTADO DO TESTE DE MCNEMAR")
    print("==================================================")
    print(f"Vitórias exclusivas da ResNet50    : {b} imagens")
    print(f"Vitórias exclusivas da MobileNetV2 : {c} imagens")
    print("--------------------------------------------------")
    print(f"Estatística Qui-Quadrado : {resultado.statistic:.4f}")
    print(f"P-Value                  : {resultado.pvalue:.4e}")
    print("==================================================")

    if resultado.pvalue < 0.05:
        conclusao = "P-value < 0.05. A superioridade da ResNet50 é ESTATISTICAMENTE SIGNIFICATIVA."
    else:
        conclusao = "P-value >= 0.05. A diferença é considerada RUÍDO AMOSTRAL (Empate Técnico)."
    
    print(conclusao)

    # 4. Salvando o log como prova documental
    caminho_log = DIRETORIO_ATUAL / "log_mcnemar.txt"
    with open(caminho_log, "w", encoding="utf-8") as f:
        f.write("RELATÓRIO DE AVALIAÇÃO ESTATÍSTICA (MCNEMAR)\n")
        f.write("Comparativo: ResNet50 vs MobileNetV2\n")
        f.write("-" * 45 + "\n")
        f.write(f"Casos em que SÓ a ResNet50 acertou   : {b}\n")
        f.write(f"Casos em que SÓ a MobileNetV2 acertou: {c}\n")
        f.write(f"Estatística Qui-Quadrado             : {resultado.statistic:.4f}\n")
        f.write(f"P-Value                              : {resultado.pvalue:.4e}\n")
        f.write("-" * 45 + "\n")
        f.write(f"Conclusão: {conclusao}\n")

    print(f"\n[SUCESSO] Prova estatística salva com sucesso em {caminho_log}")

if __name__ == "__main__":
    executar_teste_mcnemar()