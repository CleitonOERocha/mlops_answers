
##########################################################################
###### Bibliotecas -------------------------------------------------------
##########################################################################

# EDA ----------------------
import pandas as pd
from flask_sqlalchemy import SQLAlchemy

# Salvar modelo --------------
import pickle
# Visualização ---------------------
import seaborn as sns
import matplotlib.pyplot as plt
# ML -----------------------------
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
# mlflow -----------------------------
import mlflow

##########################################################################
###### Criando modelo ----------------------------------------------------
##########################################################################

# Definindo database para armazenar experimentos ------
#mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Criando/Carregando experimento ---------------------
mlflow.set_experiment("nyc-taxi-experiment")

# Função para ler e ajustar dataframe ------------------
def read_dataframe(filename):
    df = pd.read_parquet(filename, engine = 'pyarrow')

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

# Aplicando função ------------------------
df_train = read_dataframe('green_tripdata_2021-01.parquet')
df_val = read_dataframe('green_tripdata_2021-02.parquet')
df_val.head()

# Ajustando colunas -----------------------------
df_train['PU_DO'] = df_train['PULocationID'] + '_' + df_train['DOLocationID']
df_val['PU_DO'] = df_val['PULocationID'] + '_' + df_val['DOLocationID']

# Definindo colunas númericas e categoricas ---------------------
categorical = ['PU_DO'] #'PULocationID', 'DOLocationID']
numerical = ['trip_distance']

# Aplicando transformação -------------------
dv = DictVectorizer()

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dicts)

# Definindo variável target -----------------------
target = 'duration'
y_train = df_train[target].values
y_val = df_val[target].values

# Testando modelo de regressão linear -----------------------
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_val)

mean_squared_error(y_val, y_pred, squared=False)

# Salvando modelo de regressão linear -------------
with open('lin_reg.bin', 'wb') as f_out:
    pickle.dump((dv, lr), f_out)


##########################################################################
###### Usando MLflow -----------------------------------------------------
##########################################################################

with mlflow.start_run():

    # Criando tag ----------------------
    mlflow.set_tag("cientista_dados", "cleiton")
 
    # Criando parâmetros com nome dos datasets ----------------------
    mlflow.log_param("train-data-path", "./data/green_tripdata_2021-01.csv")
    mlflow.log_param("valid-data-path", "./data/green_tripdata_2021-02.csv")

    # Criando parâmetros valor de alpha na regressão de lasso ----------------------
    alpha = 0.1
    mlflow.log_param("alpha", alpha)
    lr = Lasso(alpha)
    lr.fit(X_train, y_train)

    # Criando parâmetros valor de rmse na regressão de lasso ----------------------   
    y_pred = lr.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mlflow.log_metric("rmse", rmse)

    # Criando um artefato (eles servem para reproduzir o modelo) ----------------
    mlflow.log_artifact(local_path="lin_reg.bin", artifact_path="models_pickle")