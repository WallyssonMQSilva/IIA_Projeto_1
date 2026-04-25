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

    def recomendar_por_perfil(self, tipo, combustivel, preco, top_n=5):
        """
        Recebe as preferências do usuário e retorna os N carros mais similares.
        """
        #Transforma o input do usuário na mesma estrutura de texto dos carros
        perfil_usuario = f"{tipo} {combustivel} {preco}"
        
        # Vetoriza o perfil do usuário 
        vetor_usuario = self.vetorizador.transform([perfil_usuario])
        
        # Calcula a Similaridade de Cosseno
        similaridades = cosine_similarity(vetor_usuario, self.matriz_tfidf_carros)
        
        # similaridades retorna uma matriz 2D. Pegamos o primeiro (e único) array de resultados
        scores_similaridade = similaridades[0]
        
        # Adiciona os scores ao dataframe temporariamente para ranquear
        df_resultados = self.df.copy()
        df_resultados['Score_Similaridade'] = scores_similaridade
        
        # Ordena os resultados do maior score para o menor e pega os top_N
        recomendacoes = df_resultados.sort_values(by='Score_Similaridade', ascending=False).head(top_n)
        
        # Retorna apenas as colunas relevantes
        return recomendacoes[['Modelo', 'Tipo', 'Combustivel', 'Preco', 'Score_Similaridade']]

#Teste Local

if __name__ == "__main__":
    motor = MotorTFIDF()
    print("Testando recomendação para um perfil: SUV, Híbrido, Alto\n")
    
    resultados = motor.recomendar_por_perfil("SUV", "Híbrido", "Alto")
    print(resultados)