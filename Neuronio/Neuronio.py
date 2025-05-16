import numpy as np

pesos = np.random.uniform(-0.5, 0.5, 4) # Inicialização dos pesos aleatoriamente
n = 0.3  # Taxa de aprendizado

# Definição dos dados
X = np.array([
    [-1, -0.6508, 0.1097, 4.0009],
    [-1, -1.4492, 0.8896, 4.4005],
    [-1, 2.0850, 0.6876, 12.0710],
    [-1, 0.2626, 1.1476, 7.7985],
    [-1, 0.6418, 1.0234, 7.0427],
    [-1, 0.2569, 0.6730, 8.3265],
    [-1, 1.1155, 0.6043, 7.4446],
    [-1, 0.0914, 0.3399, 7.0677],
    [-1, 0.0121, 0.5256, 4.6316],
    [-1, -0.0429, 0.4660, 5.4323],
    [-1, 0.4340, 0.6870, 8.2287],
    [-1, 0.2735, 1.0287, 7.1934],
    [-1, 0.4839, 0.4851, 7.4850],
    [-1, 0.4089, -0.1267, 5.5019],
    [-1, 1.4391, 0.1614, 8.5843],
    [-1, -0.9115, -0.1973, 2.1962],
    [-1, 0.3654, 1.0475, 7.4858],
    [-1, 0.2144, 0.7515, 7.1699],
    [-1, 0.2013, 1.0014, 6.5489],
    [-1, 0.6483, 0.2183, 5.8991],
    [-1, -0.1147, 0.2242, 7.2435],
    [-1, -0.7970, 0.8795, 3.8762],
    [-1, -1.0625, 0.6366, 2.4707],
    [-1, 0.5307, 0.1285, 5.6883],
    [-1, -1.2200, 0.7777, 1.7252],
    [-1, 0.3957, 0.1076, 5.6623],
    [-1, -0.1013, 0.5989, 7.1812],
    [-1, 2.4482, 0.9455, 11.2095],
    [-1, 2.0149, 0.6192, 10.9263],
    [-1, 0.2012, 0.2611, 5.4631],
])

Y = np.array([
    -1, -1, -1, 1, 1, -1, 1, -1, 1, 1,
    -1, 1, -1, -1, -1, -1, 1, 1, 1, 1,
    -1, 1, 1, 1, 1, -1, -1, 1, -1, 1
])

# Dividindo os dados em 2/3 para treino e 1/3 para teste
split = int(len(X) * 2 / 3)
X_train, X_teste = X[:split], X[split:]
y_train, y_teste = Y[:split], Y[split:]


def agregacao(pesos, linha):
    return np.dot(pesos, linha)

def ativacao(u):
    return 1 if u > 0 else -1 if u < 0 else 0

def reajustePeso(entrada, peso, e):
    return peso + (n * e * entrada)


meta = False
epoca = 0
max_epocas = 1000  # Limite para evitar loop infinito

while not meta and epoca < max_epocas:
    epoca += 1
    erros = 0
    
    # print(f"\nÉpoca {epoca}:")
    for i, line in enumerate(X_train):  
        Yesperado = y_train[i] 
        resAgregacao = agregacao(pesos, line)
        Yencontrado = ativacao(resAgregacao)   
        e = Yesperado - Yencontrado
        
        # print(f"\nAmostra {i}:")
        # print("  Entrada: ", line)
        # print("  Agregação: ", resAgregacao)
        # print("  Saída encontrada: ", Yencontrado)
        # print("  Saída esperada: ", Yesperado)
        # print("  Erro: ", e)
        
        if e != 0:
            erros += 1
            for j in range(len(pesos)):  
                pesos[j] = reajustePeso(line[j], pesos[j], e)

    if erros == 0:
        meta = True


print("\nFASE DE TESTES")

acertos = 0

for i, line in enumerate(X_teste):  
    Yesperado = y_teste[i] 
    resAgregacao = agregacao(pesos, line)
    Yencontrado = ativacao(resAgregacao)

    # print(f"\nAmostra {i}:")
    # print("  Entrada: ", line)
    # print("  Agregação: ", resAgregacao)
    # print("  Saída encontrada: ", Yencontrado)
    # print("  Saída esperada: ", Yesperado)
    
    if Yencontrado == Yesperado:
        acertos += 1

acuracia = acertos / len(X_teste) * 100

print("\nTreinamento concluído!")
print(f"Pesos finais: {pesos}")
print(f"Acurácia no teste: {acuracia:.2f}%")
