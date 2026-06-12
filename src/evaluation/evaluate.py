import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

def plotar_curvas_aprendizado(historico, pasta_destino):
    Path(pasta_destino).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(historico.history['accuracy'], label='Treino', linewidth=2)
    plt.plot(historico.history['val_accuracy'], label='Validação', linewidth=2)
    plt.title('Evolução da Acurácia', fontweight='bold')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(historico.history['loss'], label='Treino', linewidth=2)
    plt.plot(historico.history['val_loss'], label='Validação', linewidth=2)
    plt.title('Evolução do Erro (Loss)', fontweight='bold')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(Path(pasta_destino) / "curvas_aprendizado.png", dpi=300)
    plt.close()
    print("[SUCESSO] Curvas de aprendizado salvas em docs/imagens/")

def avaliar_modelo(modelo, ds_teste, nomes_classes, pasta_destino):
    print("\n[INFO] Iniciando Avaliação Final (Métricas e Matriz de Confusão)...")
    
    y_true = np.concatenate([y for x, y in ds_teste], axis=0)
    y_pred = np.argmax(modelo.predict(ds_teste, verbose=0), axis=1)
    
    Path(pasta_destino).mkdir(parents=True, exist_ok=True)
    
    report_dict = classification_report(y_true, y_pred, target_names=nomes_classes, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    suportes = df_report['support'].values 
    df_plot = df_report.drop(columns=['support'])

    plt.figure(figsize=(10, 5))
    ax = sns.heatmap(df_plot, annot=True, cmap="Greens", fmt=".2f", cbar=False, annot_kws={"size": 14})
    for i, suporte in enumerate(suportes):
        ax.text(3.2, i + 0.5, f"Support: {int(suporte)}", va='center', ha='left', fontsize=12, fontweight='bold')
    plt.title("Relatório de Classificação - ResNet50 Otimizada", fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(pasta_destino) / "relatorio_metricas.png", dpi=300)
    plt.close()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=nomes_classes, yticklabels=nomes_classes)
    plt.title("Matriz de Confusão", fontweight='bold')
    plt.ylabel("Classe Verdadeira")
    plt.xlabel("Predição do Modelo")
    plt.tight_layout()
    plt.savefig(Path(pasta_destino) / "matriz_confusao.png", dpi=300)
    plt.close()
    
    print("[SUCESSO] Matriz e Relatório salvos em docs/imagens/")