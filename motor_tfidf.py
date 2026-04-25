import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MotorTFIDF:
    def __init__(self, caminho_dados='carros.csv'):
        # Carrega o dataset
        self.df = pd.read_csv(caminho_dados)
        
        # Prepara o documento de cada carro
        self.df['Caracteristicas'] = self.df['Tipo'] + " " + self.df['Combustivel'] + " " + self.df['Preco']
        
        # Inicializa e treina o Vetorizador TF-IDF
        self.vetorizador = TfidfVectorizer()
        # fit_transform cria o vocabulário base com os 50 carros e já gera a matriz matemática
        self.matriz_tfidf_carros = self.vetorizador.fit_transform(self.df['Caracteristicas'])

    def calcular_scores(self, perfil_usuario):
        # Transforma o perfil do usuário em números usando o mesmo vetorizador
        vetor_usuario = self.vetorizador.transform([perfil_usuario])
        
        # Calcula a similaridade (valores de 0 a 1)
        similaridades = cosine_similarity(vetor_usuario, self.matriz_tfidf_carros).flatten()
        
        # Monta o dicionário de resposta
        dicionario_scores = {}
        for idx, linha in self.df.iterrows():
            nome_carro = linha['Modelo']
            dicionario_scores[nome_carro] = similaridades[idx]
            
        return dicionario_scores
    