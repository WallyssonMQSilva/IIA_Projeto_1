import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity


class MotorColaborativo:
    def __init__(self):
        baseDir = os.path.dirname(__file__)
        caminho = os.path.join(baseDir, 'matriz_utilidade.csv')

        self.matrizAvaliacoes = pd.read_csv(caminho, index_col=0)

        similaridade = cosine_similarity(self.matrizAvaliacoes)

        self.tabelaSimilaridade = pd.DataFrame(
            similaridade,
            index=self.matrizAvaliacoes.index,
            columns=self.matrizAvaliacoes.index
        )

    def preverNota(self, usuario, carro, k=5):
        usuariosParecidos = self.tabelaSimilaridade[usuario].drop(usuario)
        topUsuarios = usuariosParecidos.sort_values(ascending=False).head(k)

        soma = 0
        pesos = 0

        for outroUsuario, similaridade in topUsuarios.items():
            nota = self.matrizAvaliacoes.loc[outroUsuario, carro]

            if nota > 0:
                soma += nota * similaridade
                pesos += similaridade

        if pesos == 0:
            return 0

        return soma / pesos

    def recomendarCarros(self, usuario, topN=5):
        carrosNaoAvaliados = self.matrizAvaliacoes.loc[usuario]
        carrosNaoAvaliados = carrosNaoAvaliados[carrosNaoAvaliados == 0].index

        recomendacoes = {}

        for carro in carrosNaoAvaliados:
            notaPrevista = self.preverNota(usuario, carro)
            recomendacoes[carro] = notaPrevista

        recomendacoesOrdenadas = sorted(
            recomendacoes.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return recomendacoesOrdenadas[:topN]


if __name__ == "__main__":
    motor = MotorColaborativo()

    usuarioTeste = "Usuario_1"

    resultados = motor.recomendarCarros(usuarioTeste)

    print(f"\nRecomendações para {usuarioTeste}:\n")

    for carro, nota in resultados:
        print(f"{carro}: {nota:.2f}")