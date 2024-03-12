from fastapi import FastAPI, HTTPException
import model_repo_reshape
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pymongo
import uuid
import dataprocessing


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/run_process_forecast", tags=["Forecast"])
async def run_process_forecast():
    store = 137
    brand = 'tropicana'
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
    #mywrite.delete_many({})
    records = data.to_dict(orient='records')
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

    for i in range(1,8):  # 循环迭代创建和训练七个不同的 LSTM 模型
        # model_function_name = f"create_LSTM_model{i}"
        # model = model_repo_reshape.model_function_name(12,1)  # 创建模型
        model_function_name = f"create_LSTM_model{i}"
        create_model_function = getattr(model_repo_reshape, model_function_name)  # 获取模型创建函数
        model = create_model_function(12, 1)

        model_name = f'models/LSTM_model_{i}.h5'  # 模型文件名中包含索引以区分不同的模型
        es = EarlyStopping(monitor='loss', min_delta=0, patience=15, verbose=1, mode='auto')
        mc = ModelCheckpoint(filepath=model_name, save_best_only=True)
        callbacks = [es, mc]
        print("train model ",i)
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
        #print(mape_train, mape_valid, mape_test, rmse_train, rmse_valid, rmse_test, mae_train, mae_valid, mae_test, pred_train, pred_valid, pred_test)
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

    return {"message": "Task - Run forecast for active SKUs has been triggered mape is","mape": mape_test }



@app.get("/model_mape/{uuid}")
async def get_model_mape(uuid: str):
    # 在数据库中查找指定 UUID 对应的记录
    myclient = pymongo.MongoClient("mongodb://localhost:27017")
    mydb = myclient["mydb1"]
    mycol = mydb["MAPE"]  # 修改为你的集合名称
    result = mycol.find_one({"uuid": uuid})

    # 如果找不到指定 UUID 的记录，则返回 404 错误
    if result is None:
        raise HTTPException(status_code=404, detail="UUID not found")

    model_mape_list = []

    # 循环遍历1到7号模型
    for i in range(1, 8):
        # 获取当前模型的名称和 MAPE 值
        model_name_key = f"LSTM_{i}"
        mape_key_train = f"mape_train_{i}"
        mape_key_valid = f"mape_valid_{i}"
        mape_key_test = f"mape_test_{i}"
        mae_key_train = f"mae_train_{i}"
        mae_key_valid = f"mae_valid_{i}"
        mae_key_test = f"mae_test_{i}"
        rmse_key_train = f"rmse_train_{i}"
        rmse_key_valid = f"rmse_valid_{i}"
        rmse_key_test = f"rmse_test_{i}"

        mape_train = result.get(mape_key_train, f"MAPE_train {i} not found")
        mape_valid = result.get(mape_key_valid, f"MAPE_valid {i} not found")
        mape_test = result.get(mape_key_test, f"MAPE_test {i} not found")
        mae_train = result.get(mae_key_train, f"mae_train {i} not found")
        mae_valid = result.get(mae_key_valid, f"mae_valid {i} not found")
        mae_test = result.get(mae_key_test, f"mae_test {i} not found")
        rmse_train = result.get(rmse_key_train, f"rmse_train {i} not found")
        rmse_valid = result.get(rmse_key_valid, f"rmse_valid {i} not found")
        rmse_test = result.get(rmse_key_test, f"rmse_test {i} not found")

        # 构建一个字典存储当前模型的名称和 MAPE 值
        model_mape_dict = {"model_name": model_name_key, "mape_train": mape_train,"mape_valid": mape_valid,"mape_test": mape_test,"mae_train":mae_train,
                           "mae_valid":mae_valid,"mae_test":mae_test,"rmse_train":rmse_train,"rmse_valid":rmse_valid,"rmse_test":rmse_test}

        # 将当前模型的字典添加到列表中
        model_mape_list.append(model_mape_dict)

    return model_mape_list


@app.get("/get_pred_value/{uuid}/{model_no}")
async def get_pred_value(uuid: str, model_no: int):#加时间
    # 在数据库中查找指定 UUID 对应的记录
    myclient = pymongo.MongoClient("mongodb://localhost:27017")
    mydb = myclient["mydb1"]
    result = mydb["MAPE"].find_one({"uuid": uuid})

    mywrite = mydb[uuid]

    data_from_mongodb = list(mywrite.find())
    store=dataprocessing.store
    brand=dataprocessing.brand
    data = pd.DataFrame(data_from_mongodb)
    data = data[(data['store'] == store) & (data['brand'] == brand)]

    time_train, time_valid, time_test = dataprocessing.get_date(data.week)
    time_train = [int(value) for value in time_train.tolist()]  # 将 Series 中的所有值转换为整数列表
    time_valid = [int(value) for value in time_valid.tolist()]
    time_test = [int(value) for value in time_test.tolist()]


    # 如果找不到指定 UUID 的记录，则返回 404 错误
    if result is None:
        raise HTTPException(status_code=404, detail="UUID not found")

    # 获取对应模型号的预测值的键
    pred_train_key = f"pred_train_{model_no}"  # 假设预测值的键为 pred_test_i，其中 i 为模型号
    pred_valid_key = f"pred_valid_{model_no}"
    pred_test_key = f"pred_test_{model_no}"

    # 如果找不到对应模型号的预测值，则返回 404 错误
    if pred_train_key not in result:
        raise HTTPException(status_code=404, detail="Prediction value not found for the specified model number")

    # 获取对应模型号的预测值
    pred_train_value = result[pred_train_key]
    pred_valid_value = result[pred_valid_key]
    pred_test_value = result[pred_test_key]

    return {"pred_train_value": pred_train_value,"pred_valid_value": pred_valid_value,"pred_test_value": pred_test_value,"time_train":time_train,
            "time_valid":time_valid,"time_test":time_test}