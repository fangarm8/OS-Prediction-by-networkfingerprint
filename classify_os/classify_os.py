import pickle
import pandas as pd
import numpy as np
import json
import xgboost as xgb

# Обучение тестовой модели показало, что эти параметры, из ранее выбранных, являются самыми полезными
columns_to_keep = [
    "ip",
    "user_agent",
    "tls.extensions",
    "tls.ja3",
    "tls.peetprint_hash",
    "http2.akamai_fingerprint",
    "tcpip.ip.ttl",
    "tcpip.tcp.mss",
    "tcpip.tcp.window",
    #"os_prediction.highest" # целевая переменная
]

# Label encode для целевой переменной
os_dict = {'Android': np.int64(0), 'Linux': np.int64(1), 'Mac OS': np.int64(2), 'Windows': np.int64(3), 'iOS': np.int64(4)}
os_dict_decode = {np.int64(0): 'Android', np.int64(1): 'Linux', np.int64(2): 'Mac OS', np.int64(3): 'Windows', np.int64(4): 'iOS'}
all_tls_versions = ["TLS_GREASE (0xeaea)", "TLS 1.3", "TLS 1.2", "TLS 1.1", "TLS 1.0", 'TLS_GREASE (0x3a3a)', 'TLS_GREASE (0x4a4a)', 'TLS_GREASE (0x1a1a)', 'TLS_GREASE (0x5a5a)', 'TLS_GREASE (0xcaca)', 'TLS_GREASE (0x8a8a)', 'TLS_GREASE (0x9a9a)', 'TLS_GREASE (0xbaba)', 'TLS_GREASE (0xfafa)', '0300', 'TLS_GREASE (0x7a7a)', 'TLS_GREASE (0xaaaa)', 'TLS_GREASE (0xdada)', 'TLS_GREASE (0x6a6a)', 'TLS_GREASE (0x0a0a)', 'TLS_GREASE (0x2a2a)']
# Укажите свои пути к файлам
path_to_df = '../my_phone.json'
path_to_model = '../models/xgb_2.pkl'
path_to_vectorizer = '../vectorizer/tfidf_vectorizer.pkl'

def extract_supported_versions(extensions):
    if isinstance(extensions, list):
        for ext in extensions:
            if ext.get("name") == "supported_versions (43)":
                return ext.get("versions", [])
    return None

def encode_tls_versions(tls_list):
    return {version: version in tls_list for version in all_tls_versions}

def clean_data(path_to_df):
    raw_df = pd.read_json(path_to_df)
    raw_df = raw_df.to_dict(orient='records')
    raw_df_exp = pd.json_normalize(raw_df, sep='_', max_level=None)

    # Группировка по IP и проверка уникальных user_agents для каждого IP
    conflicting_ips = raw_df_exp.groupby('ip')['user_agent'].nunique()
    conflicting_ips = conflicting_ips[conflicting_ips > 1]

    # Для каждого IP, где есть несколько user_agent, оставим только первую запись
    df_cleaned = raw_df_exp[~raw_df_exp['ip'].isin(conflicting_ips.index)]
    df_cleaned = pd.concat([df_cleaned, raw_df_exp[raw_df_exp['ip'].isin(conflicting_ips.index)].drop_duplicates(subset=['ip'])])

    columns_to_keep_normalized = [col.replace(".", "_") for col in columns_to_keep]

    df = df_cleaned[columns_to_keep_normalized]

    # Создаем множество bool столбцов - объединение всех TLS версий
    expanded_df = df['tls_extensions'].apply(encode_tls_versions).apply(pd.Series)
    # Объединяем обратно с оригинальным DataFrame
    df = pd.concat([df, expanded_df], axis=1)

    # Очищаем от 'tls_extensions' и пропусков
    df = df.copy()
    df.drop(columns=['tls_extensions'], inplace=True)
    df.dropna(inplace=True)
    return df

def split_data_set(df):
    X = df.drop(columns=['os_prediction_highest'])
    y = df['os_prediction_highest'].map(os_dict)
    return X, y

def vectorize_features(df, path_to_vectorizer):
    # Векторизируем тестовые параметры, чтобы модель могла с ними работать
    if df.get('os_prediction_highest') is not None:
        X = df.drop(columns=['os_prediction_highest'])
    else:
        X = df

    X = X.drop(columns=['ip'])
    with open(path_to_vectorizer, "rb") as f:
        vectorizer = pickle.load(f)
    X_transformed = vectorizer.transform(X)
    X = pd.DataFrame(X_transformed.toarray(), columns=vectorizer.get_feature_names_out())
    X = X.drop(columns='user_agent_tfidf__x86_64')
    return X

def load_model(path_to_model):
    with open(path_to_model, 'rb') as f:
        model = pickle.load(f)
    return model

df = clean_data(path_to_df)
X = vectorize_features(df, path_to_vectorizer)
model = load_model(path_to_model)

X_dmatrix = xgb.DMatrix(X)
predictions = model.predict(X_dmatrix)
predictions = pd.Series(predictions).map(os_dict_decode)

df['os_pred'] = predictions
json_pred = df[['ip', 'os_pred']].to_dict(orient='records')

with open("predictions.json", 'w') as f:
    json.dump(json_pred, f, indent=4)