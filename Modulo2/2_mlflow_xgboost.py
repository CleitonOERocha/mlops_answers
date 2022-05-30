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
###### Pt1.: Usando MLflow -----------------------------------------------
##########################################################################

# Rodando o modelo com diversos parâmetros ----------------------

""" def objective(params):
    with mlflow.start_run():
        # Criando tag ----------------------
        mlflow.set_tag("model", "xgboost")
        # Criando parâmetros com nome dos datasets ----------------------
        mlflow.log_params(params)
        # Aplicando modelo xgboost --------------------------------
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )
        y_pred = booster.predict(valid)
        # Criando parâmetros valor de rmse ----------------------   
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

    return {'loss': rmse, 'status': STATUS_OK}


#***# Nota: Isso vem depois de rodar o código acima. O código abaixo está pegando os valores do melhor modelo acima e rodando 
search_space = {
    'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
    'learning_rate': hp.loguniform('learning_rate', -3, 0),
    'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
    'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
    'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
    'objective': 'reg:linear',
    'seed': 42
}

best_result = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=Trials()
) """

# Desabilitando o o autolog do xgboost no mlflow, apenas para fins do curso ---------------------
mlflow.xgboost.autolog(disable=True)

##########################################################################
###### Pt2.: Usando MLflow como um experimento completo ------------------
##########################################################################

# Rodando com os melhores parâmetros selecionados anteriormente no mlflow ----------------------

with mlflow.start_run():
    
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    best_params = {
        'learning_rate': 0.09585355369315604,
        'max_depth': 30,
        'min_child_weight': 1.060597050922164,
        'objective': 'reg:linear',
        'reg_alpha': 0.018060244040060163,
        'reg_lambda': 0.011658731377413597,
        'seed': 42
    }

    mlflow.log_params(best_params)

    booster = xgb.train(
        params=best_params,
        dtrain=train,
        num_boost_round=1000,
        evals=[(valid, 'validation')],
        early_stopping_rounds=50
    )

    y_pred = booster.predict(valid)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mlflow.log_metric("rmse", rmse)

    # Criando um artefato (eles servem para reproduzir o modelo) ----------------
    with open("preprocessor.b", "wb") as f_out:
        pickle.dump(dv, f_out)
    mlflow.log_artifact("preprocessor.b", artifact_path="preprocessor")

    mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
