import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

def gerar_auditoria_grad_cam(modelo, df_teste, nomes_classes, pasta_destino, diretorio_raiz):
    print("\n[INFO] Realizando Auditoria de Interpretabilidade (Grad-CAM)...")
    Path(pasta_destino).mkdir(parents=True, exist_ok=True)

    def aplicar_grad_cam(img_path, modelo, nome_classe_real):
        caminho_absoluto = str(Path(diretorio_raiz) / img_path)
        img_original = cv2.imread(caminho_absoluto)
        
        if img_original is None:
            print(f"[AVISO] Não foi possível carregar a imagem: {caminho_absoluto}")
            return
            
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        
        img_tensor = tf.image.resize(img_original, [224, 224])
        img_tensor = tf.expand_dims(img_tensor, axis=0) 
        
        ultima_camada_conv = modelo.get_layer('conv5_block3_out')
        modelo_extrator = tf.keras.Model(inputs=modelo.inputs, outputs=[modelo.output, ultima_camada_conv.output])
        
        with tf.GradientTape() as tape:
            predicoes, conv_out = modelo_extrator(img_tensor, training=False)
            indice_classe = tf.argmax(predicoes[0])
            score_classe = predicoes[:, indice_classe]
            
        gradientes = tape.gradient(score_classe, conv_out)
        pesos_gradientes = tf.reduce_mean(gradientes, axis=(0, 1, 2))
        
        mapa_calor = conv_out[0] @ pesos_gradientes[..., tf.newaxis]
        mapa_calor = tf.squeeze(mapa_calor)
        mapa_calor = tf.maximum(mapa_calor, 0) / tf.math.reduce_max(mapa_calor)
        mapa_calor = mapa_calor.numpy()
        
        mapa_calor_red = cv2.resize(mapa_calor, (img_original.shape[1], img_original.shape[0]))
        mapa_calor_red = np.uint8(255 * mapa_calor_red)
        mapa_calor_colorido = cv2.applyColorMap(mapa_calor_red, cv2.COLORMAP_JET)
        imagem_sobreposta = cv2.addWeighted(img_original, 0.6, mapa_calor_colorido, 0.4, 0)
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img_original); plt.title(f"Real: {nome_classe_real}", fontweight='bold'); plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(imagem_sobreposta); plt.title("Auditoria Grad-CAM", fontweight='bold'); plt.axis('off')
        
        plt.tight_layout()
        nome_arquivo_limpo = nome_classe_real.replace(" ", "_").lower()
        plt.savefig(Path(pasta_destino) / f"grad_cam_{nome_arquivo_limpo}.png", dpi=300)
        plt.close()

   
    for patologia in nomes_classes:
        df_patologia = df_teste[df_teste['classe'] == patologia]
        if not df_patologia.empty:
            aplicar_grad_cam(df_patologia.iloc[0]['caminho_arquivo'], modelo, patologia)

    print("[SUCESSO] Auditoria Grad-CAM concluída. Imagens salvas em docs/imagens/")