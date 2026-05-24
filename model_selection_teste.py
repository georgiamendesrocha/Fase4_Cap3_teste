import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Carregando o dataset Wine
wine = load_wine()

# X = características / dados de entrada
X = wine.data

# y = rótulos / alvo que o modelo tentaria prever
y = wine.target

# Convertendo para DataFrame para melhor visualização
df = pd.DataFrame(X, columns=wine.feature_names)
df['target'] = y

# Separando X e y
X = df.drop('target', axis=1)
y = df['target']

# Dividindo os dados em 70% treino e 30% teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# Exibindo os tamanhos dos conjuntos
print(f"Tamanho do conjunto de dados original: {X.shape}")
print(f"Tamanho do conjunto de treinamento: {X_train.shape}")
print(f"Tamanho do conjunto de teste: {X_test.shape}")

# Exibindo os primeiros valores antes e depois do split
print("\nConjunto de dados antes do split:")
print(df['target'].head().to_string())

print("\nConjunto de treinamento:")
print(y_train.head().to_string())