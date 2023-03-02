


import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import mlflow

tracking_uri = "ubuntu@ec2-18-234-211-60.compute-1.amazonaws.com"
s3_bucket = "s3://mlflow-bucket-cleiton" 
experiment_name = 'mais_um_teste'

##########################################################################
# Função para ler e ajustar dataframe ------------------
##########################################################################

def read_dataframe(filename):

    filename_parquet = filename

    df = pd.read_parquet(filename_parquet, engine = 'pyarrow')

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    print('Dataframe carregado')
    
    return df

##########################################################################
# Função para criar dados de treino e teste ------------------
##########################################################################

def tratamento_dataset(df_train, df_val):

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

    print('Dataframes de treino e teste carregados')

    return X_train, X_val, y_train, y_val

##########################################################################
# Rodando com os melhores parâmetros selecionados anteriormente no mlflow ----------------------
##########################################################################

def train_model_xgboost(X_train, X_val, y_train, y_val):
    
    mlflow.set_tracking_uri('teste_2022-10-16')

    #mlflow.set_tracking_uri(f"http://{tracking_uri}:5000")
    #client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    #expr_name = experiment_name 
    #mlflow.create_experiment(expr_name, s3_bucket)
    #mlflow.set_experiment(expr_name)
    #mlflow.get_experiment_by_name(experiment_name)

    with mlflow.start_run():
    
        print('Treinando modelos...')

        expr_name = 'teste_dia_2022-10-16_v2' 
        mlflow.create_experiment(expr_name, 'models_mlflow')
        mlflow.set_experiment(expr_name)

        train = xgb.DMatrix(X_train, label = y_train)
        valid = xgb.DMatrix(X_val, label = y_val)

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

        print('Modelos treinados')

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared = False)
        mlflow.log_metric("rmse", rmse)

        print('RMSE: ', rmse)

            # Criando um artefato (eles servem para reproduzir o modelo) ----------------
        #with open("preprocessor.b", "wb") as f_out:
        #    pickle.dump(dv, f_out)
        
        #mlflow.log_artifact("preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")


##########################################################################
# Função para fazer tudo ------------------
##########################################################################

def run_model():
    
    link_dataset_1 = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-01.parquet'
    link_dataset_2 = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-02.parquet'
   
    # Aplicando função ------------------------
    df_train = read_dataframe(link_dataset_1)
    df_val = read_dataframe(link_dataset_2)

    X_train, X_val, y_train, y_val = tratamento_dataset(df_train, df_val)

    train_model_xgboost(X_train, X_val, y_train, y_val)

    artifact_uri = mlflow.get_artifact_uri()

    print(artifact_uri)
    mlflow.end_run()


##########################################################################
###### Rodando ----------------------------------------------
##########################################################################

run_model()