import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def assess(config):

    tr = {'Accuracy'   : 'Exactitud', 
          'Precision'  : 'Precisión', 
          'Specificity': 'Especificidad', 
          'Recall'     : 'Sensibilidad',
          'Dice'       : 'Sorensen-Dice',
          'F1Score'    : 'F1 Score',
          'Jaccard'    : 'Índice Jaccard'
    }

    print(f"\nAnalisis del modelo: {config['files']['train_Loss']}\n")

    print(f'\nEntrenamiento\n')

    df_loss  = pd.read_csv(config['files']['train_Loss'], index_col=[0]) 
    df_train = pd.read_csv(config['files']['train_mets'], index_col=[0])
    df_val   = pd.read_csv(config['files']['val_mets'], index_col=[0])
    df_test  = pd.read_csv(config['files']['test_mets'], index_col=[0])

    mean_losses = df_loss.groupby("Epoca")["Loss"].mean()
    print(f'Costo inicial = {mean_losses[0]:.3f}')
    print(f'Costo final (ultima epoca) = {mean_losses[49]:.3f}')
    print(f'Diferencia = {mean_losses[0] - mean_losses[49]:.3f}\n')

    plt.plot(range(1, 51), mean_losses, marker='o')
    plt.title(f"Función de costo: {config['hyperparams']['crit']}\nÉpoca Vs. Costo")
    plt.xticks(np.arange(1, 51, step=1))
    plt.xlabel(f'Época')
    plt.ylabel(f"{config['hyperparams']['crit']}")
    #plt.show()
    plt.savefig(os.path.join(config['plots'], 'loss_plot.pdf'), dpi=300, format='pdf')

    plt.close('all')

    mets = list(df_train.columns.values)[2:-1]

    for col in mets:

        if ('F1Score' in col) or ('Jaccard' in col):
            continue

        mean_mets = df_train.groupby("Epoca")[col].mean()
        print(f'Average {tr[col]} (Train) = {mean_mets[49]:.3f}')

        plt.plot(range(1, 51), mean_mets, marker='o', label=tr[col])

    plt.title(f'Métricas promedio (Entrenamiento)')
    plt.xticks(np.arange(1, 51, step=1))
    plt.xlabel(f'Época')
    plt.ylabel(f'Valor')
    plt.legend(title='Métricas:')
    plt.savefig(os.path.join(config['plots'], f'train_metrics.pdf'), dpi=300, format='pdf')

    plt.close('all')

    #fixCSV(config['files']['val_mets'])

    print(f'\nValidacion\n')

    for col in mets:

        if ('F1Score' in col) or ('Jaccard' in col):
            continue

        mean_mets = df_val.groupby("Epoca")[col].mean()
        print(f'Average {tr[col]} (Validación) = {mean_mets[49]:.3f}')

        plt.plot(range(1, 51), mean_mets, marker='o', label=tr[col])

    plt.title(f'Métricas promedio (Validación)')
    plt.xticks(np.arange(1, 51, step=1))
    plt.yticks(np.arange(1.1, step=0.1))
    plt.xlabel(f'Época')
    plt.ylabel(f'Valor')
    plt.legend(title='Métricas:')
    plt.savefig(os.path.join(config['plots'], f'val_metrics.pdf'), dpi=300, format='pdf')

    plt.close('all')

    print(f'\nPrueba\n')

    print(df_test.columns.values)

    mets = list(df_test.columns.values)[1:-1]

    df_melted = pd.melt(df_test, id_vars=['Batch'], value_vars=mets)

    for col in mets:

        if ('F1Score' in col) or ('Jaccard' in col):
            continue

        mean_mets = df_test[col].mean()
        print(f'Average {tr[col]} (Prueba) = {mean_mets:.3f}')

    # sns.boxplot(data=df_test['Accuracy'])#, x="Batch", y="value", hue="variable")
    # plt.show()
    
    







