import model_repo_reshape
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
import pymongo
import uuid
import dataprocessing
import asyncio
from fastapi import FastAPI, HTTPException, status, Request,BackgroundTasks
import os, sys
import time

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


import subprocess


async def run_subprocess_script(name: str):
    print(f'run_subprocess_script({name}) triggered')
    start_time = time.time()

    # 使用 subprocess.Popen 创建子进程
    proc = subprocess.Popen([sys.executable, f"subprocess_{name}.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()

    end_time = time.time()
    process_time = end_time - start_time
    print(f"Subprocess {name} took {process_time:.2f} seconds")

#
# async def run_subprocess_script(name:str):
#     print(f'run_subprocess_script({name}) triggered')
#     start_time = time.time()
#     proc = await asyncio.create_subprocess_exec(
#         sys.executable, f"subprocess_{name}.py",
#         stdout=asyncio.subprocess.PIPE,
#         stderr=asyncio.subprocess.PIPE
#         )
#
#     # proc = await asyncio.create_subprocess_shell(
#     #     f"python subprocess_{name}.py",
#     #     stdout=asyncio.subprocess.PIPE,
#     #     stderr=asyncio.subprocess.PIPE
#     # )
#
#     stdout, stderr = await proc.communicate()
#     # streaming subprocess output ref: https://gist.github.com/gh640/50953484edfa846fda9a95374df57900
#
#     await proc.communicate()
#     end_time = time.time()
#     process_time = end_time - start_time
#     # Check if the log file for this task has been created
#     # files = os.listdir('./files')
#     # is_log_found = False
#     # for file in files:
#     #     file_created_time = os.path.getctime(os.path.join('./files', file))
#     #     if file.endswith('.log') and file.startswith(name) and file_created_time>start_time:
#     #         is_log_found = True
#     #         break
#     # if is_log_found:
#     #     print(f"log file - {file} was created for this task.")
#     # else:
#     #     print(f"Something went wrong. The script did not generate the output log file.")
#     print(f"Subprocess {name} took {process_time:.2f} seconds")


@app.get("/run_process_forecast", tags=["Forecast"])
async def run_process_forecast():
    asyncio.create_task(run_subprocess_script('fcast_train'))
    #BackgroundTasks.add_task(run_subprocess_script, 'fcast_train')
    return {"message": "Task - Run forecast for active SKUs has been triggered"}


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


@app.get("/get_historical_data/{uuid}")#using uuid to get historical data
async def get_historical_data(uuid: str):
    # 在数据库中查找指定 UUID 对应的记录
    myclient = pymongo.MongoClient("mongodb://localhost:27017")
    mydb = myclient["mydb1"]
    my_data = mydb[uuid]

    if my_data is None:
        raise HTTPException(status_code=404, detail="UUID not found")

    data_from_mongodb = list(my_data.find())

    store = dataprocessing.store
    brand = dataprocessing.brand
    data = pd.DataFrame(data_from_mongodb)
    data = data[(data['store'] == store) & (data['brand'] == brand)]
    week = data.week
    week = [int(value) for value in week.tolist()]
    logmove = data.logmove
    logmove = [float(value) for value in logmove.tolist()]

    return {"historical_data": logmove, "time": week}


@app.get("/check_status/{uuid}")
async def check_status(uuid: str):
    # 在数据库中查找指定 UUID 对应的记录
    myclient = pymongo.MongoClient("mongodb://localhost:27017")
    mydb = myclient["mydb1"]

    result = mydb["MAPE"].find_one({"uuid": uuid})
    if result is None:
        raise HTTPException(status_code=404, detail="UUID not found")

    status_key = f"status"
    status = result.get(status_key, f"status not found")

    return {"status : ",status}