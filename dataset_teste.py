

import pandas as pd

# Criando o conjunto de dados fictício
data = {
    'Cultura': ['Milho', 'Soja', 'Café', 'Cana-de-açúcar', 'Arroz', 'Algodão'],
    'Região': ['Norte', 'Sul', 'Sudeste', 'Nordeste', 'Centro-Oeste', 'Nordeste'],
    'Produção (toneladas)': [5000, 12000, 8000, 15000, 11000, 7000],
    'Preço por Tonelada (R$)': [750, 1300, 2000, 600, 900, 1400],
    'Tipo de Solo': ['Argiloso', 'Arenoso', 'Misto', 'Argiloso', 'Arenoso', 'Misto']
}

# Convertendo em DataFrame
df = pd.DataFrame(data)

# Exibindo o dataset no terminal
print(df.head())



from sklearn.preprocessing import LabelEncoder

# Inicializando o LabelEncoder
label_encoder = LabelEncoder()

# Aplicando Label Encoding na coluna 'Cultura'
df['Cultura_LabelEncoded'] = label_encoder.fit_transform(df['Cultura'])

# Exibindo no terminal
print(df[['Cultura', 'Cultura_LabelEncoded']])

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Inicializando o OneHotEncoder
onehot_encoder = OneHotEncoder(sparse_output=False)

# Aplicando One-Hot Encoding na coluna 'Região'
onehot_encoded = onehot_encoder.fit_transform(df[['Região']])

# Convertendo para DataFrame
encoded_columns = onehot_encoder.get_feature_names_out(['Região'])

onehot_df = pd.DataFrame(
    onehot_encoded,
    columns=encoded_columns
)

# Juntando os resultados ao DataFrame original
df = pd.concat([df, onehot_df], axis=1)

# Exibindo no terminal
print(df[['Região', 'Região_Centro-Oeste', 'Região_Nordeste', 'Região_Norte', 'Região_Sul', 'Região_Sudeste']].to_string(index=False))

from sklearn.preprocessing import OrdinalEncoder

# Aplicando Ordinal Encoding na coluna 'Tipo de Solo'
ordinal_encoder = OrdinalEncoder(
    categories=[['Arenoso', 'Argiloso', 'Misto']]
)

df['TipoSolo_OrdinalEncoded'] = ordinal_encoder.fit_transform(df[['Tipo de Solo']])

# Exibindo no terminal
print(df[['Tipo de Solo', 'TipoSolo_OrdinalEncoded']].to_string(index=False))