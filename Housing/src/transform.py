import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib 
import os

def imput_transform(df, imputer_dir='../imputers'):
    imputers = joblib.load(f'{imputer_dir}/imputers.pkl')
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(imputers[col])
            if df[col].isna().any():
                raise Exception(f'{col}: unable to find a matching imputer')
    return df

# 1. impute and save imputers
# 2. encode categorical features
# 3. cluster code specific numerical features
def impute_fit_transform(df, imputer_dir='../imputers'):
    # impute a column if it has null values 
    imputers = {}
    for col in df.columns:
        if df[col].dtype == 'object': #categorical
            mode = df[col].mode()[0]
            imputers[col] = mode
            if df[col].isna().any():
                df[col] = df[col].fillna(mode)
            
        else: # numerical
            med = df[col].median()
            imputers[col] = med
            if df[col].isna().any():
                df[col] = df[col].fillna(med)
    joblib.dump(imputers, f'{imputer_dir}/imputers.pkl')
    return df 

def cluster_transform(df, features_to_cluster, n_clusters=7, scaler_dir='../Scalers', kmeans_dir='../Kmeans'):
    # Remove any duplicated features
    features_to_cluster = list(set(features_to_cluster))

    scaler_models = joblib.load(f'{scaler_dir}/cluster_scaler.pkl')
    kmeans_models = joblib.load(f'{kmeans_dir}/cluster_kmeans.pkl')

    for feature_name in features_to_cluster:
        # 1. scale using the stored scaler 
        scaler = scaler_models[feature_name]
        scaled_feature = scaler.transform(df[[feature_name]])

        # 2. cluster using stored kmeans 
        kmeans = kmeans_models[feature_name]
        df[f'cluster_{feature_name}'] = kmeans.predict(scaled_feature)
    # 3. drop original features 
    df = df.drop(columns=features_to_cluster)
    return df

def cluster_fit_transform(df, features_to_cluster, n_clusters=7, scaler_dir='../Scalers', kmeans_dir='../Kmeans'):
    # features_to_cluster = [
    #     'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'LotFrontage', 'LotArea', 'MasVnrArea',
    #     'BsmtFinSF1', 'BsmtFinSF2', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GarageArea',
    #     'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal'
    # ]

    # Remove any duplicated features
    features_to_cluster = list(set(features_to_cluster))

    # Ensure save directories exist
    os.makedirs(scaler_dir, exist_ok=True)
    os.makedirs(kmeans_dir, exist_ok=True)

    # Initiate scaler and kmeans dictionaries
    scaler_models = {}
    kmean_models = {}

    for feature_name in features_to_cluster:
        # 1. Scale the feature
        scaler = StandardScaler()
        # scaler.fit_transform expects a 2D dataframe with 1 column [[]]
        scaled_feature = scaler.fit_transform(df[[feature_name]])

        # 2. Cluster the scaled feature
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df[f'cluster_{feature_name}'] = kmeans.fit_predict(scaled_feature)

        # 3. Save scaler and KMeans model
        scaler_models[feature_name] = scaler 
        kmean_models[feature_name] = kmeans 

    joblib.dump(scaler_models, f'{scaler_dir}/cluster_scaler.pkl')
    joblib.dump(kmean_models, f'{kmeans_dir}/cluster_kmeans.pkl')

    # 4. Drop the original features
    df = df.drop(columns=features_to_cluster)

    return df

def encode_transform(df, labelEncoding=False, encoder_dir='../encoders'):
    categorical_features = []
    for col in df.columns:
        if df[col].dtype == 'object': #categorical
            categorical_features.append(col)

    if labelEncoding:
        encoder = joblib.load(os.path.join(encoder_dir, 'ordinal_encoder.pkl'))
        df[categorical_features] = pd.DataFrame(
            encoder.transform(df[categorical_features]),
            columns=categorical_features,
            index=df.index
        )
    
    else:
        encoder = joblib.load(os.path.join(encoder_dir, 'onehot_encoder.pkl'))
        encoded_array = encoder.transform(df[categorical_features])
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=encoder.get_feature_names_out(categorical_features),
            index=df.index
        )
        df = df.drop(columns=categorical_features)
        df = pd.concat([df, encoded_df], axis=1)
    return df                      

def encode_fit_transform(df, labelEncoding=False, encoder_dir='../encoders'):

    os.makedirs(encoder_dir, exist_ok=True)  # Ensure the encoder directory exists
    categorical_features = []
    for col in df.columns:
        if df[col].dtype == 'object': #categorical
            categorical_features.append(col)

    if labelEncoding:
        encoder = OrdinalEncoder()
        df[categorical_features] = pd.DataFrame(
            encoder.fit_transform(df[categorical_features]),
            columns=categorical_features,
            index=df.index
        )
        joblib.dump(encoder, os.path.join(encoder_dir, 'ordinal_encoder.pkl'))

    else:
        encoder = OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False)
        encoded_array = encoder.fit_transform(df[categorical_features])
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=encoder.get_feature_names_out(categorical_features),
            index=df.index
        )
        df = df.drop(columns=categorical_features)
        df = pd.concat([df, encoded_df], axis=1)

        joblib.dump(encoder, os.path.join(encoder_dir, 'onehot_encoder.pkl'))

    return df

def mutual_information_transform(df, mi_dir='../mi'):

    low_mi_features = joblib.load(os.path.join(mi_dir, 'low_mi_features.pkl'))
    interaction_features_list = joblib.load(os.path.join(mi_dir, 'interaction_features.pkl'))

    interaction_features = []
    for interaction_feature in interaction_features_list:
        feature1, feature2 = interaction_feature 
        new_feature = df[feature1] * df[feature2]   # new feature nd array 
        interaction_features.append(pd.Series(new_feature, name=f"{feature1}_{feature2}"))  # Store new feature / same interaction feature set as in training
    df = pd.concat([df] + interaction_features, axis=1)

    # drop low mi score features - must drop the same features as in training   
    df = df.drop(columns=low_mi_features)
    return df 

# mutual information analysis on all features and drop low scores 
# create interaction features and retain higher scoring features 
# prerequisite:  All features must be first encoded
# TODO: interaction features are not used - use them 
def mutual_information_fit_transform(df, y, mi_dir='../mi'):

    os.makedirs(mi_dir, exist_ok=True)
    all_features = df.columns
    # mutual information analysis on encoded df 
    mi_scores = mutual_info_regression(df, y)
    mi_scores = pd.Series(mi_scores, index=all_features).sort_values(ascending=False)

    mi_scores_df = mi_scores.reset_index()
    mi_scores_df.columns = ['Feature', 'MI_Score']

    # discover interaction features and incorporate them
    low_mi_features = mi_scores_df[mi_scores_df['MI_Score'] <= 0.01].iloc[:,0]

    # dump low scoring features
    joblib.dump(low_mi_features, os.path.join(mi_dir, 'low_mi_features.pkl'))

    print(f'features with low scores: {low_mi_features}')
    print(f'features with low scores count: {len(low_mi_features)}')
    print(f'total df columns: {len(df.columns)}')

    interaction_features = []
    interaction_features_list = []

    for i in range(len(low_mi_features)):
        for j in range(i+1, len(low_mi_features)):
            feature1 = low_mi_features.iloc[i]
            feature2 = low_mi_features.iloc[j]
            new_feature = df[feature1] * df[feature2]   # new feature nd array 
            new_feature_df = pd.DataFrame(new_feature, columns=[f'{feature1}_{feature2}'])
            mi_score = mutual_info_regression(new_feature_df, y)
            if mi_score[0] > 0.01:
                interaction_features.append(pd.Series(new_feature, name=f"{feature1}_{feature2}"))  # Store new feature
                interaction_features_list.append((feature1, feature2))

    if interaction_features:
        print(f'total interaction features: {len(interaction_features)}')
        df = pd.concat([df] + interaction_features, axis=1)
        print(f'total df columns: {len(df.columns)}')

        # dump interaction feature list 
        joblib.dump(interaction_features_list, os.path.join(mi_dir, 'interaction_features.pkl'))

    # drop low mi features 
    df = df.drop(columns=low_mi_features)
    print(f'total df columns: {len(df.columns)}')
    return df 

def save_features(features, features_dir='../features'):
    os.makedirs(features_dir, exist_ok=True)

    joblib.dump(features, os.path.join(features_dir, 'features.pkl'))

def get_features(features_dir='../features'):
    return joblib.load(os.path.join(features_dir, 'features.pkl'))

