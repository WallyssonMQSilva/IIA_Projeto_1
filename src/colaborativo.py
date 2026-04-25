import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

matrizAvaliacoes = pd.read_csv(
    r"C:\Users\vitor\IIA_Projeto_1\dados\matriz_utilidade.csv",
    index_col=0
)

similaridadeUsuarios = cosine_similarity(matrizAvaliacoes)

tabelaSimilaridade = pd.DataFrame(
    similaridadeUsuarios,
    index=matrizAvaliacoes.index,
    columns=matrizAvaliacoes.index
)


def preverNota(usuario, carro, matrizAvaliacoes, tabelaSimilaridade, k=5):
    usuariosParecidos = tabelaSimilaridade[usuario].drop(usuario)
    topUsuarios = usuariosParecidos.sort_values(ascending=False).head(k)

    soma = 0
    pesos = 0

    for outroUsuario, similaridade in topUsuarios.items():
        nota = matrizAvaliacoes.loc[outroUsuario, carro]

        if nota > 0:
            soma += nota * similaridade
            pesos += similaridade

    if pesos == 0:
        return 0

    return soma / pesos


def recomendarCarros(usuario, matrizAvaliacoes, tabelaSimilaridade, quantidade=5):
    carrosNaoAvaliados = matrizAvaliacoes.loc[usuario]
    carrosNaoAvaliados = carrosNaoAvaliados[carrosNaoAvaliados == 0].index

    recomendacoes = {}

    for carro in carrosNaoAvaliados:
        notaPrevista = preverNota(usuario, carro, matrizAvaliacoes, tabelaSimilaridade)
        recomendacoes[carro] = notaPrevista

    recomendacoesOrdenadas = sorted(
        recomendacoes.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return recomendacoesOrdenadas[:quantidade]


usuarioTeste = "Usuario_1"

recomendados = recomendarCarros(
    usuarioTeste,
    matrizAvaliacoes,
    tabelaSimilaridade
)

for carro, nota in recomendados:
    print(f"{carro}: {nota:.2f}")