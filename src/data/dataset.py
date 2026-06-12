import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

def carregar_dados(caminho_treino, caminho_teste, mapa_classes, seed, batch_size=32):
    df_treino_completo = pd.read_csv(caminho_treino)
    df_teste = pd.read_csv(caminho_teste)
    
    for df in [df_treino_completo, df_teste]:
        df["caminho_arquivo"] = df["caminho_arquivo"].str.replace('\\', '/')
        df["label"] = df["classe"].map(mapa_classes)
        
    df_treino, df_val = train_test_split(df_treino_completo, test_size=0.2, random_state=seed, stratify=df_treino_completo["label"])
    
    def processar_imagem_bruta(caminho, rotulo):
        img = tf.io.read_file(caminho)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [224, 224])
        return img, rotulo

    AUTOTUNE = tf.data.AUTOTUNE
    cam_tr = df_treino["caminho_arquivo"].str.encode('utf-8').values
    cam_val = df_val["caminho_arquivo"].str.encode('utf-8').values
    cam_te = df_teste["caminho_arquivo"].str.encode('utf-8').values
    
    ds_treino = tf.data.Dataset.from_tensor_slices((cam_tr, df_treino["label"].values)).map(processar_imagem_bruta, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    ds_val = tf.data.Dataset.from_tensor_slices((cam_val, df_val["label"].values)).map(processar_imagem_bruta, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    ds_teste = tf.data.Dataset.from_tensor_slices((cam_te, df_teste["label"].values)).map(processar_imagem_bruta, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    
    return ds_treino, ds_val, ds_teste, df_teste