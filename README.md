# Time-series-forecasting
Using MongoDB as database and Fastapi to achieve functions 

model_repo contains 8 models include DCNN and 7 different models

api_test now include 3 api, run_process_forecast first read csv file and store it into database, then do the training and evalution from begining and store the performance into datebase as well. get_model_mape using uuid to find the model performance. get_pred_value using uuid and model no. to find the pred_value and time.

dataprocessing.py contains some functions to process and evalate the date and model.
