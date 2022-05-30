##########################################################################
###### Bibliotecas -------------------------------------------------------
##########################################################################

# EDA ----------------------
import pandas as pd

# Salvar modelo --------------
import pickle

# Visualização ---------------------
import seaborn as sns
import matplotlib.pyplot as plt

# ML -----------------------------
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import LinearSVR
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

# mlflow -----------------------------
from flask_sqlalchemy import SQLAlchemy
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

# Dividindo em treino e teste -------------------------
train = xgb.DMatrix(X_train, label=y_train)
valid = xgb.DMatrix(X_val, label=y_val)

##########################################################################
###### Usando MLflow -----------------------------------------------------
##########################################################################

# Aplicando mlflow em diversos algoritmos de aprendizado ----------------

mlflow.sklearn.autolog()

for model_class in (RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, LinearSVR):

    with mlflow.start_run():

        # Criando parâmetro com nome do dataset ------------------
        mlflow.log_param("train-data-path", "green_tripdata_2021-01.csv")
        mlflow.log_param("valid-data-path", "green_tripdata_2021-02.csv")

        # Criando um artefato (eles servem para reproduzir o modelo) ----------------
        mlflow.log_artifact("preprocessor.b", artifact_path="preprocessor")

        mlmodel = model_class()
        mlmodel.fit(X_train, y_train)

        y_pred = mlmodel.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)