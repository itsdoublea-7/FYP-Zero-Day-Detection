import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import helper
from datetime import datetime           
import tensorflow as tf

def create_lstm_model(input_shape, num_features):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=64, input_shape=input_shape, return_sequences=True),
        tf.keras.layers.LSTM(units=64, return_sequences=False),
        tf.keras.layers.Dense(units=num_features)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

label_encoder_1 = preprocessing.LabelEncoder()
label_encoder_2 = preprocessing.LabelEncoder()
label_encoder_3 = preprocessing.LabelEncoder()
one_hot_encoder = preprocessing.OneHotEncoder()

def read_kdd_dataset(path):
    global label_encoder_1, label_encoder_2, label_encoder_3, one_hot_encoder
    
    dataset = pd.read_csv(path, header=None)
    if '+' not in path:
        dataset[41] = dataset[41].str[:-1]

    dataset[42] = ''
    dataset = dataset.values
    dataset = helper.add_kdd_main_classes(dataset)
    
    if hasattr(label_encoder_1, 'classes_') == False:
        dataset[:, 1] = label_encoder_1.fit_transform(dataset[:, 1])
        dataset[:, 2] = label_encoder_2.fit_transform(dataset[:, 2])        
        dataset[:, 3] = label_encoder_3.fit_transform(dataset[:, 3])
        dataset_features = one_hot_encoder.fit_transform(dataset[:, :-2]).toarray() 
    else:
        dataset[:, 1] = label_encoder_1.transform(dataset[:, 1])
        dataset[:, 2] = label_encoder_2.transform(dataset[:, 2])        
        dataset[:, 3] = label_encoder_3.transform(dataset[:, 3])
        dataset_features = one_hot_encoder.transform(dataset[:, :-2]).toarray() 
        
    return dataset, dataset_features

def evaluate_model(model, X, y, metrics=["mse", "mae", "f1_score"]):
  predictions = model.predict(X)

  results = {}
  for metric in metrics:
    if metric == "mse":
      results["mse"] = mean_squared_error(y, predictions)
    elif metric == "mae":
      results["mae"] = mean_absolute_error(y, predictions)
    elif metric == "f1_score":
      results["f1_score"] = f1_score(y, predictions)
    else:
      raise ValueError(f"Unknown metric: {metric}")

  return results

if __name__ == '__main__':
   
    columns_to_drop = ['ip_src', 'ip_dst']

    parser = argparse.ArgumentParser()
    parser.add_argument('--normal_path', default='DataFiles/CIC/biflow_Monday-WorkingHours_Fixed.csv')
    parser.add_argument('--attack_paths', default='DataFiles/CIC/')
    parser.add_argument('--output', default='Results.csv')
    parser.add_argument('--epochs',type=int, default=50)
    parser.add_argument('--archi', default='U15,U9,U15')
    parser.add_argument('--regu', default='l2')
    parser.add_argument('--l1_value',type=float, default=0.01)
    parser.add_argument('--l2_value',type=float, default=0.0001)
    parser.add_argument('--correlation_value',type=float, default=0.9)
    parser.add_argument('--dropout',type=float, default=0.05)
    parser.add_argument('--model')
    parser.add_argument('--nu',type=float, default=0.01)
    parser.add_argument('--kern', default='rbf')
    parser.add_argument('--loss', default='mse')
    
    
    args = parser.parse_args()
    output_file = args.output

    output_file = datetime.now().strftime("%d_%m_%Y__%H_%M_") + output_file
    helper.file_write_args(args, output_file)    
    
    if('dataset_path' in args):
        dataset, dataset_features = read_kdd_dataset(args.dataset_path)
         
        normal = dataset_features[dataset[:, 42] == 'normal', :]
        
        standard_scaler = preprocessing.StandardScaler()
        normal = pd.DataFrame(standard_scaler.fit_transform(normal))
        
        if 'Train+' in args.dataset_path:
            dataset_testing, dataset_features_testing = read_kdd_dataset(args.dataset_path.replace('Train+', 'Test+'))
            dataset_features_testing = standard_scaler.transform(dataset_features_testing)
    else:
        normal = pd.read_csv(args.normal_path)
        normal = normal.dropna()
        normal.drop(columns_to_drop, axis=1, inplace=True)
        normal.drop(normal.columns[0], axis=1, inplace=True)
        normal, to_drop = helper.dataframe_drop_correlated_columns(normal, args.correlation_value)
        
        standard_scaler = preprocessing.StandardScaler()
        x_scaled = standard_scaler.fit_transform(normal.values)
        normal = pd.DataFrame(x_scaled)
        print(normal.columns)
    categorical_columns = [1, 2, 3]
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(sparse=False))])

    preprocessor = ColumnTransformer(transformers=[('cat', categorical_transformer, categorical_columns)], remainder='passthrough')

    # ...

    # Apply preprocessing to your data
    normal = preprocessor.fit_transform(normal)

    train_X, valid_X, train_ground, valid_ground = train_test_split(normal,normal, test_size=0.25, random_state=1)
    train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
    valid_X = np.reshape(valid_X, (valid_X.shape[0], 1, valid_X.shape[1]))

    lstm_model = create_lstm_model(input_shape=(1, train_X.shape[2]), num_features=train_X.shape[2])

    history = lstm_model.fit(train_X, train_ground, epochs=args.epochs, batch_size=32, validation_data=(valid_X, valid_ground))

    lstm_model.save('lstm_model.h5')
    
    validation_results = evaluate_model(lstm_model, valid_X, valid_ground, metrics=["mse", "mae", "f1_score"])
    # Print the evaluation results.
    print("Validation results:")
    for metric, value in validation_results.items():
        print(f"{metric}: {value}")
