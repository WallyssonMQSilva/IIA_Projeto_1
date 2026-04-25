import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RecomendadorColaborativo:
    def __init__(self, caminho_csv="matriz_utilidade.csv"):
        # Carrega a matriz definindo a primeira coluna (Usuários) como índice
        self.df_matriz = pd.read_csv(caminho_csv, index_col=0)
        
        # Preenche possíveis valores vazios com 0
        self.df_matriz = self.df_matriz.fillna(0)
        self.nomes_carros = self.df_matriz.columns.tolist()

    def prever_notas(self, dicionario_notas_usuario):
        # Cria o vetor do novo usuário (tamanho 50)
        vetor_novo_usuario = np.zeros(len(self.nomes_carros))
        for i, carro in enumerate(self.nomes_carros):
            if carro in dicionario_notas_usuario:
                vetor_novo_usuario[i] = dicionario_notas_usuario[carro]
                
        vetor_novo_usuario = vetor_novo_usuario.reshape(1, -1)
        
        # Proteção contra o "Início Frio" total (usuário não avaliou nada ainda)
        if np.sum(vetor_novo_usuario) == 0:
            return {carro: 0.0 for carro in self.nomes_carros}

        # Calcula a similaridade entre o novo usuário e a base de treinamento
        similaridades = cosine_similarity(vetor_novo_usuario, self.df_matriz.values).flatten()
        
        # Previsão usando Média Ponderada
        dicionario_scores = {}
        soma_similaridades = np.sum(np.abs(similaridades))
        
        if soma_similaridades == 0:
             return {carro: 0.0 for carro in self.nomes_carros}

        for i, carro in enumerate(self.nomes_carros):
            if carro in dicionario_notas_usuario:
                # Se ele já avaliou na interface, apenas divide por 5 para normalizar (0 a 1)
                dicionario_scores[carro] = dicionario_notas_usuario[carro] / 5.0
            else:
                # Multiplica as notas dos outros usuários pela similaridade com o novo usuário
                notas_outros = self.df_matriz.iloc[:, i].values
                nota_prevista = np.dot(similaridades, notas_outros) / soma_similaridades
                
                # Divide por 5 para normalizar (0 a 1)
                dicionario_scores[carro] = nota_prevista / 5.0
                
        return dicionario_scores
