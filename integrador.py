from motor_tfidf import MotorTFIDF
from colaborativo import RecomendadorColaborativo

class GerenciadorHibrido:
    def __init__(self):
        # Inicializa os dois motores passando os arquivos CSV
        self.motor_cont = MotorTFIDF("carros.csv")
        self.motor_colab = RecomendadorColaborativo("matriz_utilidade.csv")

    def obter_recomendacao_final(self, perfil_usuario, notas_usuario, peso_tfidf=0.5):
        #=Busca scores de cada modelo
        scores_cont = self.motor_cont.calcular_scores(perfil_usuario)
        scores_colab = self.motor_colab.prever_notas(notas_usuario)
        
        ranking_hibrido = {}
        peso_colab = 1.0 - peso_tfidf
        
        # Média ponderada
        for carro in scores_cont.keys():
            s1 = scores_cont[carro]
            s2 = scores_colab.get(carro, 0)
            
            # Cálculo final do Score Híbrido
            ranking_hibrido[carro] = (s1 * peso_tfidf) + (s2 * peso_colab)
            
        # ordenação final
        return sorted(ranking_hibrido.items(), key=lambda x: x[1], reverse=True)[:5]


#teste local
if __name__ == "__main__":
    import os

    print("Iniciando teste local do Integrador Híbrido...\n")

    # Verifica se os arquivos CSV estão na pasta
    if not os.path.exists("carros.csv") or not os.path.exists("matriz_utilidade.csv"):
        print("❌ Erro: Os arquivos 'carros.csv' e/ou 'matriz_utilidade.csv' não foram encontrados.")
    else:
        # 1. Instancia o integrador
        print("⏳ Carregando os motores (TF-IDF e Colaborativo)...")
        integrador = GerenciadorHibrido()
        print("✅ Motores carregados com sucesso!\n")

        # 2. Define os dados de teste (juntando os dois testes anteriores)
        perfil_teste = "SUV Gasolina Alto"
        notas_teste = {
            "Toyota Hilux": 5,
            "Ford Ranger": 5,
            "Renault Kwid": 1,
            "Fiat Mobi": 1
        }
        
        print(f"👤 Perfil em Texto: '{perfil_teste}'")
        print(f"⭐ Notas Dadas: {notas_teste}\n")

        # --- TESTE 1: 100% TF-IDF ---
        print("▶️ TESTE 1: Peso 100% TF-IDF (Ignora as notas)")
        resultado_tfidf = integrador.obter_recomendacao_final(perfil_teste, notas_teste, peso_tfidf=1.0)
        for i, (carro, score) in enumerate(resultado_tfidf, 1):
            print(f"   {i}. {carro} - Score: {score:.4f}")
            
        # --- TESTE 2: 100% Colaborativo ---
        print("\n▶️ TESTE 2: Peso 100% Colaborativo (Ignora o texto e foca nos vizinhos)")
        resultado_colab = integrador.obter_recomendacao_final(perfil_teste, notas_teste, peso_tfidf=0.0)
        for i, (carro, score) in enumerate(resultado_colab, 1):
            print(f"   {i}. {carro} - Score: {score:.4f}")

        # --- TESTE 3: 50% / 50% ---
        print("\n▶️ TESTE 3: Modelo Híbrido (Equilíbrio 50/50)")
        resultado_hibrido = integrador.obter_recomendacao_final(perfil_teste, notas_teste, peso_tfidf=0.5)
        for i, (carro, score) in enumerate(resultado_hibrido, 1):
            print(f"   {i}. {carro} - Score: {score:.4f}")

        print("\n🏁 Teste finalizado!")