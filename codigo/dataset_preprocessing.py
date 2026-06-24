import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df_original = pd.read_csv('Pokemon.csv')
df_processado = df_original.copy()

print("--- Dataset Original Carregado ---")
print(df_processado.head(3))

# Pré-Processamento
# Primeiro substituímos os valores nulos da coluna 'type2' por 'None'
df_processado['type2'] = df_processado['type2'].fillna('None')

# Remover colunas e problemas irrelevantes que não ajudam o modelo a prever padrões ('Number' e 'Name')
df_processado = df_processado.drop(columns=['number', 'name'])

# Converter a classe 'legendary' de True/False para 1/0
df_processado['legendary'] = df_processado['legendary'].astype(int)

# Aplicar One-Hot Encoding nas colunas categóricas (type1 e type2)
# Isso vai criar colunas como type1_Fire, type1_Water, etc., cheias de 0 e 1
df_processado = pd.get_dummies(df_processado, columns=['type1', 'type2'], dtype=int)

# Normalizando os dados fazendo uma lista das colunas numéricas de status que queremos colocar na escala 0 a 1
colunas_status = ['total', 'hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']

# Inicializar o escalador
scaler = MinMaxScaler()

# Normalizar as colunas de status
df_processado[colunas_status] = scaler.fit_transform(df_processado[colunas_status])

# Salvar o processo em um novo arquivo CSV
df_processado.to_csv('pokemon_processed.csv', index=False)

print("\n--- Pré-processamento Concluído ---")
print(f"Colunas originais: {df_original.shape[1]} -> Colunas pós-encoding: {df_processado.shape[1]}")
print("Arquivo 'pokemon_processed.csv' gerado.")