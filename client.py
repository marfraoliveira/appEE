import os
import requests
import json
from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler
#%%
import numpy as np
import pandas as pd
#Gerando dados para inserir no modelo (API)
def createDf():
    columns = ['x','y','z']
    c = np.random.rand(90,3)
    df = pd.DataFrame(c,columns=columns)
    df.to_json('txtDf.txt')

createDf()




#%%

# URL da rota no servidor Flask
urlL = 'http://localhost:5000/api'  # Substitua pelo endereço correto
urlR = 'https://apprahdl.onrender.com/api'

# Abre o arquivo JSON em modo de leitura
    # Carrega o conteúdo do arquivo JSON em um objeto Python

# Agora, 'data' contém o conteúdo do arquivo JSON como um dicionário ou uma lista
#print(data)


# Dados que você deseja enviar em formato JSON

with open('txtDf.txt', 'r') as json_file:
    data = json.load(json_file)
    


# Especifique o caminho completo para o arquivo que deseja excluir
arquivo_para_excluir = "./resultPredict.csv"

# Verifique se o arquivo existe antes de tentar excluí-lo
if os.path.exists(arquivo_para_excluir):
    # Exclua o arquivo
    os.remove(arquivo_para_excluir)
    print(f"O arquivo {arquivo_para_excluir} foi excluído com sucesso.")
else:
    print(f"O arquivo {arquivo_para_excluir} não existe.")


# Realiza a solicitação POST enviando os dados JSON
csv_handler = CSVHandler('resultPredict.csv')
@measure_energy(handler=csv_handler)
def predict():
    response = requests.post(urlL, json=data)
# Verifica a resposta do servidor Flask
    if response.status_code == 200:
       result = response.json()
       print('Resultado da Predição:', str(result)  )
    else:
       print('Erro ao fazer a solicitação:', response.status_code)

for _ in range(100):
    predict()


csv_handler.save_data()



#%% Calcular média do resultado do consumo de energia

import pandas as pd

def calcular_media_consumo_energia(csv_filename):
    # Lê o arquivo CSV em um DataFrame
    try:
        df = pd.read_csv(csv_filename, delimiter=';')
    except FileNotFoundError:
        return "Arquivo não encontrado."
    except Exception as e:
        return f"Erro ao ler o arquivo CSV: {str(e)}"

    # Verifica se a coluna 'package_0' existe no DataFrame
    if 'package_0' not in df.columns:
        return "A coluna 'package_0' não existe no arquivo CSV."

    # Calcula a média do campo 'package_0'
    media_consumo_energia = df['package_0'].mean()

    return media_consumo_energia

# Exemplo de uso
csv_filename = './resultPredict.csv'  # Substitua pelo nome do seu arquivo CSV
media = calcular_media_consumo_energia(csv_filename)

if isinstance(media, str):
    print(media)
else:
    print(f"A média do consumo de energia (package_0) é: {media}")

#%%

import pandas as pd
columns = ['timestamp','tag','duration','package_0','core_0','uncore_0']
# Carregar o arquivo CSV em um DataFrame
df = pd.read_csv('resultPredict.csv',sep=';')

data = pd.DataFrame(df,columns=columns)

# Especificar o número de períodos para a média móvel
periodos = 5  # Você pode ajustar esse valor conforme necessário

# Calcular a média móvel para o campo 'package_0'
df['MediaMovel_package_0'] = df['package_0'].rolling(window=periodos).mean()

# Imprimir o DataFrame com a média móvel
print(df)

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carregar o arquivo CSV em um DataFrame
df = pd.read_csv('resultPredict.csv',sep=';')

# Especificar o número de períodos para a média móvel
periodos = 5  # Você pode ajustar esse valor conforme necessário

# Calcular a média móvel para o campo 'package_0'
df['MediaMovel_package_0'] = df['package_0'].rolling(window=periodos).mean()

# Converter a coluna 'timestamp' para um NumPy array
timestamps = df['timestamp'].values
package_0 = df['package_0'].values
media_movel = df['MediaMovel_package_0'].values

#%% Plotar o gráfico
plt.figure(figsize=(12, 6))
plt.plot(timestamps, package_0, label='package_0', color='blue')
plt.plot(timestamps, media_movel, label=f'Média Móvel ({periodos} períodos)', color='red')
plt.xlabel('Timestamp')
plt.ylabel('Valor')
plt.legend()
plt.title('Gráfico com Média Móvel para package_0')
plt.grid(True)

# Formatando os rótulos do eixo x para melhor legibilidade
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Converter a coluna 'timestamp' para um NumPy array
timestamps = pd.to_datetime(df['timestamp']).values
package_0 = df['package_0'].values

# Plotar o gráfico apenas com os valores da coluna 'package_0'
plt.figure(figsize=(12, 6))
plt.plot(timestamps, package_0, label='package_0', color='blue')
plt.xlabel('Timestamp')
plt.ylabel('Valor')
plt.legend()
plt.title('Gráfico para package_0')
plt.grid(True)
plt.xticks(rotation=45)  # Rotacionar os rótulos do eixo x para melhor legibilidade
plt.tight_layout()
plt.show()



#%%
import pandas as pd
import matplotlib.pyplot as plt


# Calcular a média, mediana e moda do campo 'package_0'
media_package_0 = df['package_0'].mean()
mediana_package_0 = df['package_0'].median()
moda_package_0 = df['package_0'].mode().values[0]

# Plotar o gráfico
plt.figure(figsize=(8, 6))
plt.hist(df['package_0'], bins=30, alpha=0.5, color='blue', label='package_0')
plt.axvline(media_package_0, color='red', linestyle='dashed', linewidth=2, label='Média')
plt.axvline(mediana_package_0, color='green', linestyle='dashed', linewidth=2, label='Mediana')
plt.axvline(moda_package_0, color='purple', linestyle='dashed', linewidth=2, label='Moda')
plt.xlabel('Valor')
plt.ylabel('Frequência')
plt.legend()
plt.title('Histograma com Média, Mediana e Moda para package_0')
plt.grid(True)
plt.show()




