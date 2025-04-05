# This script simulates "daily retraining" of the model (`current_model.keras`)
# using a data from a REST API.

# It saves versioned copies of the model and preprocessing pipeline
# and updates the current active model (current_model.keras) and transformer (transform_pipeline.pkl) for prediction use.


# ======================================
#       Simulate REST API 
# ======================================

# response = requests.get("https://name.com/api/data")
# url = 
# df = pd.read_csv(url)


# ====================================
#      Data preprocessing
# =====================================



df = pd.read_csv("data/share_data_today.csv")
df_clean = df.dropna()

df = df.drop([
    'ch9__f_7', 'ch9__f_8', 'ch9__f_9', 'ch9__f_10', 'ch9__f_11',
], axis=1)


df['timeslot_datetime_from'] = pd.to_datetime(df['timeslot_datetime_from'])
df["hour"] = pd.to_datetime(df["timeslot_datetime_from"]).dt.hour
df["day"] = pd.to_datetime(df["timeslot_datetime_from"]).dt.day_name()

df["day"] = pd.Categorical(
    df["timeslot_datetime_from"].dt.day_name(),
    categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    ordered=True
)

# =====================================
#   train-test split
# =====================================

X_train_full, X_test, y_train_full, y_test = train_test_split(
    df_clean.drop(columns=['share_15_54', 'timeslot_datetime_from', 'main_ident', 'date', 'share_15_54_3mo_mean']),
    df_clean['share_15_54'],
    test_size=0.2,
    random_state=42
)
X_train_full.shape, X_test.shape, y_train_full.shape, y_test.shape    


X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full,
    y_train_full,
    random_state=42
)

# ================
#    Pipeline
# ================

scaler = StandardScaler()
one_hot_encoder = OneHotEncoder(handle_unknown='ignore') 


num_vars = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_vars = X_train.select_dtypes(include=['object']).columns


transform_pipeline = ColumnTransformer([
    ('scaler', scaler, num_vars),
    ('one_hot_encoder', one_hot_encoder, cat_vars),
])


X_train_transformed = transform_pipeline.fit_transform(X_train)
X_test_transformed = transform_pipeline.transform(X_test)
X_valid_transformed = transform_pipeline.transform(X_valid)




