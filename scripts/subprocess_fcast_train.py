import dataprocessing
import pymongo
import uuid
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import time
import model_repo_reshape
import logging
import datetime

start_time = time.time()
# Set up the logging configuration

store = dataprocessing.store
brand = dataprocessing.brand
myclient = pymongo.MongoClient("mongodb://localhost:27017")
mydb = myclient["mydb1"]  # 指定数据库
myuuid = mydb["MAPE"]

# generate UUID
uuid_value = str(uuid.uuid4())
# write the uuid into database
myuuid.insert_one({"uuid": uuid_value})

csv_file = "data/Orange Juice.csv"
data = pd.read_csv(csv_file)

mywrite = mydb[uuid_value]

# put data into MongoDB
records = data.to_dict(orient='records')

result = myuuid.find_one({"uuid": uuid_value})
if result is not None:
    if "status" in result:
        # 如果已经存在状态字段，则更新状态
        if result["status"] != "upload":
            myuuid.update_one({"uuid": uuid_value}, {"$set": {"status": "upload"}})
    else:
        # 如果不存在状态字段，则添加状态
        myuuid.update_one({"uuid": uuid_value}, {"$set": {"status": "upload"}})

mywrite.insert_many(records)
print("successfully write into database")

data_from_mongodb = list(mywrite.find())
data = pd.DataFrame(data_from_mongodb)
data = data[(data['store'] == store) & (data['brand'] == brand)]

scaler = MinMaxScaler(feature_range=(0, 1))
data['Normalized_logmove'] = scaler.fit_transform(data['logmove'].values.reshape(-1, 1)).flatten()

X, y, y_date, start_values = dataprocessing.create_dataset(data[['Normalized_logmove']])

X_train, X_valid, X_test, y_train, y_valid, y_test, time_train, time_valid, time_test = dataprocessing.split_train_valid_test(
    X,
    y, data.week)

print(X_train.shape, X_valid.shape, X_test.shape)
print(y_train.shape, y_valid.shape, y_test.shape)
print(time_train.shape, time_valid.shape, time_test.shape)

result = myuuid.find_one({"uuid": uuid_value})
if result is not None:
    if "status" in result:
        # 如果已经存在状态字段，则更新状态
        if result["status"] != "training":
            myuuid.update_one({"uuid": uuid_value}, {"$set": {"status": "training"}})
    else:
        # 如果不存在状态字段，则添加状态
        myuuid.update_one({"uuid": uuid_value}, {"$set": {"status": "training"}})

for i in range(1, 8):  # 循环迭代创建和训练七个不同的 LSTM 模型
    model_function_name = f"create_LSTM_model{i}"
    create_model_function = getattr(model_repo_reshape, model_function_name)  # 获取模型创建函数
    model = create_model_function(12, 1)

    model_name = f'models/LSTM_model_{i}.h5'  # 模型文件名中包含索引以区分不同的模型
    es = EarlyStopping(monitor='loss', min_delta=0, patience=15, verbose=1, mode='auto')
    mc = ModelCheckpoint(filepath=model_name, save_best_only=True)
    callbacks = [es, mc]
    print("train model ", i)

    fit = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=200,
        verbose=2,
        validation_data=(X_valid, y_valid),
        callbacks=callbacks)
    mape_train, mape_valid, mape_test, rmse_train, rmse_valid, rmse_test, mae_train, mae_valid, mae_test, pred_train, pred_valid, pred_test = dataprocessing.evaluate_model(
        data, scaler, X_train,
        X_valid, X_test, y_train,
        y_valid, y_test, start_values,
        model)

    # print(mape_train, mape_valid, mape_test, rmse_train, rmse_valid, rmse_test, mae_train, mae_valid, mae_test, pred_train, pred_valid, pred_test)
    myuuid.update_one({"uuid": uuid_value}, {"$set": {
        f"mape_train_{i}": mape_train,
        f"mape_valid_{i}": mape_valid,
        f"mape_test_{i}": mape_test,
        f"rmse_train_{i}": rmse_train,
        f"rmse_valid_{i}": rmse_valid,
        f"rmse_test_{i}": rmse_test,
        f"mae_train_{i}": mae_train,
        f"mae_valid_{i}": mae_valid,
        f"mae_test_{i}": mae_test,
        f"pred_train_{i}": pred_train.tolist(),
        f"pred_valid_{i}": pred_valid.tolist(),
        f"pred_test_{i}": pred_test.tolist()
    }})

result = myuuid.find_one({"uuid": uuid_value})
if result is not None:
    if "status" in result:
        # 如果已经存在状态字段，则更新状态
        if result["status"] != "finish":
            myuuid.update_one({"uuid": uuid_value}, {"$set": {"status": "finish"}})
    else:
        # 如果不存在状态字段，则添加状态
        myuuid.update_one({"uuid": uuid_value}, {"$set": {"status": "finish"}})

logging.info(f"[INFO] PROCESS END after {(time.time() - start_time):.3f} seconds")
print(f" --- PROCESS END --- {(time.time() - start_time):.3f} seconds ---")
