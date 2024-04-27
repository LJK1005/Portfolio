import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor, RandomForestClassifier, RandomForestRegressor
from copy import deepcopy as dc
from datetime import datetime as dt
import os, pickle, io
import warnings
warnings.filterwarnings('ignore')
from streamlit_package import *

import sys
sys.path.append("Y:\Python\Mega_IT")
from yjk_module.dl_module import *

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam, Nadam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


plt.rcParams["font.family"] = 'Malgun Gothic'
plt.rcParams["font.size"] = 6
plt.rcParams["figure.figsize"] = (4, 3)
plt.rcParams["axes.unicode_minus"] = False

is_available = False

st.title("머신러닝")
st.write("머신러닝으로 모델을 학습하고 결과를 예측합니다.")

st.header("데이터 불러오기")
st.write("이전 페이지에서 진행한 데이터 입력 및 전처리 데이터를 사용하거나 전처리가 완료된 데이터를 업로드 할 수 있습니다.")
load_how = st.radio("데이터를 불러오는 방법을 선택하세요.",
                    ["이전 단계에서 전처리를 수행함", "파일 업로드"],
                    captions = ["", "csv와 xlsx 파일만 지원합니다."])

if load_how == "파일 업로드":
    uploaded_file = st.file_uploader("파일 업로드")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            is_available = True
            st.header("종속변수 컬럼과 분석 타입 선택")
            yname = st.selectbox("종속변수 컬럼 선택", df.columns)
            ml_type = st.radio("분석 타입 선택",
                    ["회귀", "분류"], captions = ["회귀분석을 실행합니다.", "분류분석을 실행합니다."])
        except:
            try:
                df = pd.read_excel(uploaded_file)
                is_available = True
                st.header("종속변수 컬럼과 분석 타입 선택")
                yname = st.selectbox("종속변수 컬럼 선택", df.columns)
                ml_type = st.radio("분석 타입 선택",
                    ["회귀", "분류"], captions = ["회귀분석을 실행합니다.", "분류분석을 실행합니다."])
            except:
                st.write("올바른 csv나 xlsx 파일을 업로드하세요.")
else:
    try:
        df = st.session_state['df_preprocess']
        yname = st.session_state['yname']
        ml_type = st.session_state['ml_type']
        is_available = True
    except:
        st.write("데이터 입력 혹은 전처리 과정을 진행하지 않았습니다. 해당 작업 수행후 진행해주세요.")

if is_available:
    st.markdown("- 불러온 데이터프레임")
    st.dataframe(df)

    st.markdown(f"- 머신러닝 유형 : {ml_type}")
    st.divider()

    st.title("머신러닝 방법 선택")
    ml_method = st.radio("머신러닝을 수행할 방법을 선택하세요.",
                         ["단일 모델 학습", "여러 모델 비교 및 앙상블" ,"딥러닝"],
                         captions = ["단일 모델에 대한 학습과 하이퍼파라미터 튜닝을 진행합니다.", "여러 모델을 학습하고 결과를 비교하며 필요에 따라 앙상블 모델을 구현합니다.", "Keras를 사용합니다."])
    
    if ml_method == "단일 모델 학습":
        st.divider()
        st.title("학습 모델 선택")
        if ml_type == "회귀":
            support = YjkRegressorSupport(alert = False)
            models = support.regressor_li
        else:
            support = YjkClassifierSupport(alert = False)
            models = support.classifier_li

        model_select = st.selectbox("학습할 모델을 선택하세요.", models)

        st.divider()
        st.header("파라미터 선택")
        st.write("추천 파라미터를 기반으로 GridSearchCV를 수행하거나 파라미터를 직접 입력할 수 있습니다.")

        param_how = st.radio("파라미터 설정 방법 선택", ["추천 파라미터로 GridSearchCV 진행", "파라미터 직접 입력"])

        if param_how == "파라미터 직접 입력":
            param_ex = support.model_dict[model_select].copy()
            for i in param_ex.index:
                param_ex.loc[i, "파라미터 범위"] = str(param_ex.loc[i, "파라미터 범위"])
            st.markdown("- 파라미터 목록 및 추천값")
            st.dataframe(param_ex)

            param_df = support.model_dict[model_select].copy()
            param_df.drop(["파라미터 범위", "파라미터 노트"], axis = 1, inplace = True)
            param_df["입력 파라미터"] = None

            st.write("파라미터를 직접 입력하고 파라미터 입력 버튼을 클릭하면 모델 생성이 완료됩니다.")
            param_df_2 = st.data_editor(param_df, disabled = ['파라미터명'])

            if st.button("파라미터 입력"):
                try:
                    params = {}
                    for i in param_df_2.index:
                        params[i] = param_df_2.loc[i, "입력 파라미터"]
                    model = support.call_model(model_select, param_show = False)

                    for i in params.keys():
                        if params[i] == "None":
                            params[i] = None
                        else:
                            try:
                                params[i] = float(params[i])
                                if params[i] - int(params[i]) == 0:
                                    params[i] = int(params[i])
                            except:
                                pass
                    
                    model.set_params(**params)
                    st.session_state["model"] = dc(model)
                    st.session_state["model2"] = dc(model)
                    st.markdown("- 입력할 파라미터")
                    st.write(params)
                    st.write("모델 생성 완료")
                    ml_done = False
                    st.session_state["ml_done"] = ml_done
                except:
                    st.write("입력한 파라미터가 적합한 값이 아닙니다. 입력값을 다시 확인하세요.")

            st.divider()

            st.title("훈련 / 검증 데이터 분리")
            st.write("훈련 / 검증데이터의 비율을 정하고 데이터를 분리합니다.")
            test_size = st.slider("검증 데이터 비율 입력", 0.01, 0.99, 0.2)
            random_seed = int(st.number_input("훈련 / 검증데이터 분리에 사용할 랜덤 시드값을 입력합니다.", value = 0, step = 1, format = "%d"))

            X = df.copy()
            Y = X.pop(yname)
            x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state = random_seed, test_size = test_size)

            st.divider()
            try:
                ml_done = st.session_state["ml_done"]
            except:
                ml_done = False
            st.title("머신러닝 수행 및 결과 확인")
            st.write("버튼 클릭 시 머신러닝을 진행합니다.")
            if st.button("실행") or ml_done:
                if ml_done:
                    model = st.session_state["model"]
                else:
                    model = st.session_state["model"]
                    model.fit(x_train, y_train)
                if ml_type == "분류":
                    if max(y_train.nunique(), y_test.nunique()) > 2:
                        is_binary = 'macro'
                    else:
                        is_binary = 'binary'
                    result_df = pd.DataFrame([
                        ["훈련 데이터", accuracy_score(y_train, model.predict(x_train)), recall_score(y_train, model.predict(x_train), average = is_binary), precision_score(y_train, model.predict(x_train), average = is_binary)],
                        ["검증 데이터", accuracy_score(y_test, model.predict(x_test)), recall_score(y_test, model.predict(x_test), average = is_binary), precision_score(y_test, model.predict(x_test), average = is_binary)]
                    ], columns = ["데이터", "Accuracy", "Recall", "Precision"]).set_index("데이터")
                    st.write("모델 훈련이 완료되었습니다.")
                    st.dataframe(result_df)
                    ml_done = True
                    st.session_state["ml_done"] = ml_done
                    st.session_state["result_df"] = result_df
                    if st.button("훈련 초기화", type = "primary"):
                        st.session_state["ml_done"] = False
                else:
                    result_df = pd.DataFrame([
                        ["훈련 데이터", r2_score(y_train, model.predict(x_train)), mean_absolute_error(y_train, model.predict(x_train)), np.sqrt(mean_squared_error(y_train, model.predict(x_train)))],
                        ["검증 데이터", r2_score(y_test, model.predict(x_test)), mean_absolute_error(y_test, model.predict(x_test)), np.sqrt(mean_squared_error(y_train, model.predict(x_train)))]
                    ], columns = ["데이터", "R2_Score", "MAE", "RMSE"]).set_index("데이터")
                    st.write("모델 훈련이 완료되었습니다.")
                    st.dataframe(result_df)
                    ml_done = True
                    st.session_state["ml_done"] = ml_done
                    st.session_state["result_df"] = result_df
                    if st.button("훈련 초기화", type = "primary"):
                        st.session_state["ml_done"] = False
                    st.download_button(
                        label="모델 객체 다운로드",
                        data=pickle.dumps(model),
                        file_name='model.pkl'
                    )

            st.divider()
            st.title("하이퍼파라미터 튜닝")
            st.write("모델에 다른 파라미터를 적용하여 결과를 비교합니다. 회귀분석의 경우 R2 Score, 분류분석의 경우 정확도를 기준으로 비교합니다.")
            
            tune_param_name = st.selectbox("튜닝할 파리미터 선택", param_ex.index)
            param_select = st.radio("입력할 파라미터 선택", ["추천 파라미터", "파라미터 직접 입력"], captions = ["", "입력할 파라미터는 리스트 형태로 직접 입력합니다."])

            if param_select == "추천 파라미터":
                tune_param = support.model_dict[model_select].loc[tune_param_name, "파라미터 범위"]
                st.write(f"비교할 파라미터 : {tune_param}")
            else:
                tune_param = st.text_input("파라미터를 리스트 형태로 입력하세요.")
                try:
                    tune_param = eval(tune_param)
                    if type(tune_param) != list:
                        st.write("입력한 값이 리스트 형태인지 확인하세요.")
                    else:
                        st.write(f"비교할 파라미터 : {tune_param}")
                except:
                    pass
                    
            if st.button("파라미터 비교 실행"):
                if ml_type == "회귀":
                    fig, tune_df = yjk_regression_param_plot(st.session_state["model2"], x_train, y_train, param_name = tune_param_name, param_list = tune_param, x_test = x_test, y_test = y_test,
                                                             figsize = (8, 4), for_st = True)
                    st.pyplot(fig, use_container_width = False, clear_figure = True)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(tune_df)
                    with col2:
                        train_score = result_df.loc["훈련 데이터", "R2_Score"]
                        test_score = result_df.loc["검증 데이터", "R2_Score"]
                        st.write(f"{tune_param_name}에 입력했던 파라미터는 {model.get_params()[tune_param_name]}입니다.")
                        st.write(f"훈련 데이터 R2 Score : {round(train_score, 4)}")
                        st.write(f"검증 데이터 R2 Score : {round(test_score, 4)}")
                else:
                    fig, tune_df = yjk_classification_param_plot(st.session_state["model2"], x_train, y_train, param_name = tune_param_name, param_list = tune_param, x_test = x_test, y_test = y_test,
                                                             figsize = (8, 4), for_st = True)
                    st.pyplot(fig, use_container_width = False, clear_figure = True)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(tune_df)
                    with col2:
                        train_score = result_df.loc["훈련 데이터", "Accuracy"]
                        test_score = result_df.loc["검증 데이터", "Accuracy"]
                        st.write(f"{tune_param_name}에 입력했던 파라미터는 {model.get_params()[tune_param_name]}입니다.")
                        st.write(f"훈련 데이터 R2 Score : {round(train_score, 4)}")
                        st.write(f"검증 데이터 R2 Score : {round(test_score, 4)}")
        else:
            param_ex = support.model_dict[model_select].copy()
            for i in param_ex.index:
                param_ex.loc[i, "파라미터 범위"] = str(param_ex.loc[i, "파라미터 범위"])
            st.markdown("- 파라미터 목록 및 추천값")
            st.dataframe(param_ex)

            is_grid = st.radio("GridSearchCV와 RandomizedSearchCV중 선택", ["GridSearchCV", "RandomizedSearchCV"])
            cv = int(st.number_input("cv 값을 입력합니다.", value = 10, step = 1, format = "%d"))
            if is_grid == "GridSearchCV":
                is_rand = False
            else:
                rand_iter = int(st.number_input("RandomizedSearchCV에 사용할 n_iter 값을 입력합니다.", value = 50, step = 1, format = "%d"))
                is_rand = True

            model = support.call_model(model_select, param_show = False)

            st.divider()

            st.title("훈련 / 검증 데이터 분리")
            st.write("훈련 / 검증데이터의 비율을 정하고 데이터를 분리합니다.")
            test_size = st.slider("검증 데이터 비율 입력", 0.01, 0.99, 0.2)
            random_seed = int(st.number_input("훈련 / 검증데이터 분리에 사용할 랜덤 시드값을 입력합니다.", value = 0, step = 1, format = "%d"))

            X = df.copy()
            Y = X.pop(yname)
            x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state = random_seed, test_size = test_size)

            st.divider()
            
            st.title("모델 학습")
            st.write("가능한 파라미터 조합에 대하여 학습을 시작합니다.")
            try:
                grid = st.session_state["grid"]
                grid_done = True
            except:
                grid_done = False

            if st.button("학습 시작"):
                if is_grid:
                    grid = GridSearchCV(model, param_grid = support.get_params(), n_jobs = int(os.cpu_count()/2), cv = cv)
                    grid.fit(x_train, y_train)
                    st.write("학습 완료")
                    st.session_state["grid"] = grid
                    grid_done = True
                else:
                    grid = RandomizedSearchCV(model, param_distributions = support.get_params(), n_jobs = int(os.cpu_count()/2), cv = cv, n_iter = rand_iter)
                    grid.fit(x_train, y_train)
                    st.write("학습 완료")
                    st.session_state["grid"] = grid
                    grid_done = True
            
            if grid_done:
                st.divider()
                st.title("학습 결과 보고")
                st.markdown("- 성능이 가장 우수한 파라미터 조합")
                st.write(grid.best_params_)
                if ml_type == "분류":
                    if max(y_train.nunique(), y_test.nunique()) > 2:
                        is_binary = 'macro'
                    else:
                        is_binary = 'binary'
                    st.markdown("- 성능 평가")
                    result_df = pd.DataFrame([
                        ["훈련 데이터", accuracy_score(y_train, grid.predict(x_train)), recall_score(y_train, grid.predict(x_train), average = is_binary), precision_score(y_train, grid.predict(x_train), average = is_binary)],
                        ["검증 데이터", accuracy_score(y_test, grid.predict(x_test)), recall_score(y_test, grid.predict(x_test), average = is_binary), precision_score(y_test, grid.predict(x_test), average = is_binary)]
                    ], columns = ["데이터", "Accuracy", "Recall", "Precision"]).set_index("데이터")
                    st.dataframe(result_df)
                else:
                    st.markdown("- 성능 평가")
                    result_df = pd.DataFrame([
                        ["훈련 데이터", r2_score(y_train, grid.predict(x_train)), mean_absolute_error(y_train, grid.predict(x_train)), np.sqrt(mean_squared_error(y_train, grid.predict(x_train)))],
                        ["검증 데이터", r2_score(y_test, grid.predict(x_test)), mean_absolute_error(y_test, grid.predict(x_test)), np.sqrt(mean_squared_error(y_test, grid.predict(x_test)))]
                    ], columns = ["데이터", "R2_Score", "MAE", "RMSE"]).set_index("데이터")
                    st.dataframe(result_df)

                    st.download_button(
                        label="Grid 객체 다운로드",
                        data=pickle.dumps(grid),
                        file_name='grid.pkl'
                    )
    elif ml_method == "여러 모델 비교 및 앙상블":
        st.divider()
        st.title("학습 모델 선택")
        if ml_type == "분류":
            support = YjkClassifierSupport(alert = False)
            models = support.classifier_li
        else:
            support = YjkRegressorSupport(alert = False)
            models = support.regressor_li
        model_li = st.multiselect("학습할 모델을 2개 이상 선택하세요.", models)
        if len(model_li) > 1:
            st.divider()
            st.title("모델 학습 준비")
            st.write("각 모델에 사용되는 하이퍼 파라미터는 추천 파라미터 기반으로 진행됩니다.")
            is_grid = st.radio("GridSearchCV와 RandomizedSearchCV중 선택", ["GridSearchCV", "RandomizedSearchCV"])
            cv = int(st.number_input("cv 값을 입력합니다.", value = 10, step = 1, format = "%d"))
            if is_grid == "GridSearchCV":
                is_rand = False
                rand_iter = 0
            else:
                rand_iter = int(st.number_input("RandomizedSearchCV에 사용할 n_iter 값을 입력합니다.", value = 50, step = 1, format = "%d"))
                is_rand = True
            if ml_type == "분류":
                out_proba = st.checkbox("Soft Voting을 위하여 SVC 모델에 predict_proba가 가능하도록 설정, 대신 학습 속도가 느려집니다.")
            
            st.divider()

            st.title("훈련 / 검증 데이터 분리")
            st.write("훈련 / 검증데이터의 비율을 정하고 데이터를 분리합니다.")
            test_size = st.slider("검증 데이터 비율 입력", 0.01, 0.99, 0.2)
            random_seed = int(st.number_input("훈련 / 검증데이터 분리에 사용할 랜덤 시드값을 입력합니다.", value = 0, step = 1, format = "%d"))

            X = df.copy()
            Y = X.pop(yname)
            x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state = random_seed, test_size = test_size)

            st.divider()

            try:
                result_df = st.session_state["grid_df"]
                grid_done = True
            except:
                grid_done = False

            st.title("모델 학습 및 결과 출력")
            if st.button("학습 시작"):
                with st.spinner("학습 진행중..."):
                    if ml_type == "분류":
                        result_df = yjk_classifier_multi_gridsearch(support, x_train, y_train, x_test, y_test, randomized_search = is_rand, randomized_iter = rand_iter, cv = cv,
                                                                    secondary_score = ['recall', 'precision'], include_models = model_li, for_st = True, include_proba = out_proba)
                    else:
                        result_df = yjk_regressor_multi_gridsearch(support, x_train, y_train, x_test, y_test, randomized_search = is_rand, randomized_iter = rand_iter, cv = cv, primary_score = 'rmse',
                                                                    secondary_score = ['r2', 'mae'], include_models = model_li, for_st = True)
                st.success("모델 학습 완료!")
                grid_done = True
                st.session_state["grid_df"] = result_df

            if grid_done:
                if ml_type == "분류":
                    result_show = result_df[['accuracy_train', 'accuracy_test', 'recall_train', 'recall_test', 'precision_train', 'precision_test', 'best_param']]
                else:
                    result_show = result_df[['r2_train', 'r2_test', 'mae_train', 'mae_test', 'rmse_train', 'rmse_test', 'best_param']]
                st.dataframe(result_show)
                
                st.header("파라미터 출력")
                st.write("모델별로 성능이 우수한 파라미터를 출력합니다.")
                param_out = st.selectbox("모델을 선택하세요.", result_df.index)
                if param_out:
                    st.write(result_df.loc[param_out, "best_param"])
                
                st.download_button(
                    label="학습 결과 다운로드",
                    data=pickle.dumps(result_df),
                    file_name='result_df.pkl'
                )

                st.divider()

                st.title("앙상블 모델 구현")
                st.write("학습한 모델들을 이용하여 앙상블 모델을 구현합니다. 앙상블 모델이 단일 모델보다 성능이 우수하다고 보장할 수는 없습니다.")

                ensemble_method = st.radio("앙상블 방법을 선택하세요.", ["Voting", "Bagging", "Stacking"],
                                            captions = ["Voting 앙상블을 구현합니다. 분류 모델의 경우 Hard Voting과 Soft Voting을 선택할 수 있습니다.",
                                                        "Bagging 앙상블을 구현합니다.",
                                                        "Stacking 앙상블을 구현합니다. Stacking에 사용될 모델은 랜덤포레스트를 이용합니다."])
                
                if ensemble_method == 'Bagging':
                    bagging_model = st.selectbox("Bagging 앙상블을 수행할 모델을 선택하세요.", result_df.index)
                    n_estimator = int(st.number_input("Bagging에 사용할 모델 갯수를 입력합니다.", value = 10, step = 1, format = "%d"))
                    bootstrap = st.toggle("Bootstrap", value = True, help = "중복 샘플링을 허용합니다. 활성화하면 Bagging, 비활성화시 Pasting으로 앙상블을 구현합니다.")
                    bagging_random = int(st.number_input("샘플 추출에 사용할 랜덤 시드 값을 입력합니다.", value = 0, step = 1, format = "%d"))

                    try:
                        ensemble = st.session_state["ensemble"]
                        ensemble_done = True
                    except:
                        ensemble_done = False

                    if st.button("앙상블 학습"):
                        if ml_type == "분류":
                            ensemble = BaggingClassifier(dc(result_df.loc[bagging_model, "best_model"]), n_estimators = n_estimator, bootstrap = bootstrap, random_state = bagging_random)
                        else:
                            ensemble = BaggingRegressor(dc(result_df.loc[bagging_model, "best_model"]), n_estimators = n_estimator, bootstrap = bootstrap, random_state = bagging_random)

                        with st.spinner("학습 진행중..."):
                            ensemble.fit(x_train, y_train)
                            st.success("앙상블 학습 완료!")
                            ensemble_done = True
                            st.session_state["ensemble"] = ensemble


                else:
                    ensemble_models = st.multiselect("앙상블 모델에 추가할 모델을 2개 이상 선택하세요.", result_df.index)
                    ensemble_model_li = []
                    for i in ensemble_models:
                        ensemble_model_li.append((i, dc(result_df.loc[i, "best_model"])))
                
                    if ensemble_method == "Voting" and ml_type == "분류":
                        is_soft = st.toggle("Soft Voting", value = True, help = "Soft Voting을 구현합니다. 모든 모델이 predict_proba를 지원해야 합니다. 일반적으로 Hard Voting보다 성능이 우수합니다.")
                        if is_soft:
                            class_method = 'soft'
                        else:
                            class_method = 'hard'
                    try:
                        ensemble = st.session_state["ensemble"]
                        ensemble_done = True
                    except:
                        ensemble_done = False

                    if st.button("앙상블 학습"):
                        if ensemble_method == "Voting":
                            if ml_type == "분류":
                                ensemble = VotingClassifier(ensemble_model_li, voting = class_method)
                            else:
                                ensemble = VotingRegressor(ensemble_model_li)
                        elif ensemble_method == "Stacking":
                            if ml_type == "분류":
                                ensemble = StackingClassifier(ensemble_model_li, RandomForestClassifier())
                            else:
                                ensemble = StackingRegressor(ensemble_model_li, RandomForestRegressor())
                        with st.spinner("학습 진행중..."):
                            ensemble.fit(x_train, y_train)
                            st.success("앙상블 학습 완료!")
                            ensemble_done = True
                            st.session_state["ensemble"] = ensemble
                    
                if ensemble_done:
                    st.header("앙상블 성능 확인")
                    ensemble_vs = "Accuracy" if ml_type == "분류" else "R2 Score"
                    st.write(f"단일 모델과의 비교는 {ensemble_vs}가 가장 우수한 모델과 수행합니다.")

                    if ml_type == "분류":
                        if max(y_train.nunique(), y_test.nunique()) > 2:
                            is_binary = 'macro'
                        else:
                            is_binary = 'binary'
                        ensemble_result = pd.DataFrame([
                            [accuracy_score(y_train, ensemble.predict(x_train)), accuracy_score(y_test, ensemble.predict(x_test)),
                            recall_score(y_train, ensemble.predict(x_train), average = is_binary), recall_score(y_test, ensemble.predict(x_test), average = is_binary),
                            precision_score(y_train, ensemble.predict(x_train), average = is_binary), precision_score(y_test, ensemble.predict(x_test), average = is_binary)]
                        ], columns = ['accuracy_train', 'accuracy_test', 'recall_train', 'recall_test', 'precision_train', 'precision_test'], index = ["ensemble"])
                        ensemble_result.loc["single"] = result_df.iloc[0, 2:]
                        ensemble_result.columns = ['Accuracy_train', 'Accuracy_test', 'Recall_train', 'Recall_test', 'Precision_train', 'Precision_test']
                    else:
                        ensemble_result = pd.DataFrame([
                            [r2_score(y_train, ensemble.predict(x_train)), r2_score(y_test, ensemble.predict(x_test)),
                            mean_absolute_error(y_train, ensemble.predict(x_train)), mean_absolute_error(y_test, ensemble.predict(x_test)),
                            np.sqrt(mean_squared_error(y_train, ensemble.predict(x_train))), np.sqrt(mean_squared_error(y_test, ensemble.predict(x_test)))]
                        ], columns = ['r2_train', 'r2_test', 'mae_train', 'mae_test', 'rmse_train', 'rmse_test'], index = ["ensemble"])
                        ensemble_result.loc["single"] = result_df.iloc[0, 2:]
                        ensemble_result.columns = ['R2_train', 'R2_test', 'MAE_train', 'MAE_test', 'RMSE_train', 'RMSE_test']
                    deltas = {}
                    for i in ensemble_result.columns:
                        deltas[i] = ensemble_result.loc["ensemble", i] - ensemble_result.loc["single", i]

                    st.dataframe(ensemble_result)

                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    cols = [col1, col2, col3, col4, col5, col6]

                    if ml_type == "분류":
                        better_li = ["normal", "normal", "normal", "normal", "normal", "normal"]
                    else:
                        better_li = ["normal", "normal", "inverse", "inverse", "inverse", "inverse"]

                    for i, v in enumerate(cols):
                        with v:
                            st.metric(list(deltas.keys())[i], value = "%.4f" % ensemble_result.iloc[0, i], delta = "%.4f" % deltas[list(deltas.keys())[i]], delta_color = better_li[i])

                    st.download_button(
                        label="앙상블 모델 다운로드",
                        data=pickle.dumps(ensemble),
                        file_name='ensemble.pkl'
                    )
    if ml_method == "딥러닝":
        st.divider()

        st.title("훈련 / 검증 데이터 분리")
        st.write("훈련 / 검증데이터의 비율을 정하고 데이터를 분리합니다.")
        test_size = st.slider("검증 데이터 비율 입력", 0.01, 0.99, 0.2)
        random_seed = int(st.number_input("훈련 / 검증데이터 분리에 사용할 랜덤 시드값을 입력합니다.", value = 0, step = 1, format = "%d"))

        X = df.copy()
        Y = X.pop(yname)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state = random_seed, test_size = test_size)

        st.divider()
        st.title("Keras 인공신경망 구성")
        if ml_type == '분류':
            st.write("마지막 층은 Dense 층으로 고정됩니다. 종속변수의 unique 수에 따라 마지막 층의 활성화 함수는 시그모이드 혹은 소프트맥스로 자동 결정됩니다.")
            n_y = max(y_train.nunique(), y_test.nunique())
            if n_y > 2:
                st.write(f"종속변수는 총 {n_y}종으로 softmax 함수를 사용합니다.")
                final_activation = 'softmax'
                final_units = n_y
            else:
                st.write(f"종속변수는 총 {n_y}종으로 sigmoid 함수를 사용합니다.")
                final_activation = 'sigmoid'
                final_units = 1
        else:
            st.write("마지막 층은 1단 Dense로 고정됩니다. 활성화 함수는 linear를 사용합니다.")
            final_activation = 'linear'
            final_units = 1

        st.header("추가할 신경망 층 선택")

        if st.button("초기화", type = 'primary'): 
            df_dense = pd.DataFrame()
            st.session_state['keras_df'] = df_dense
            nth = 1
            st.session_state['nth'] = nth

        try:
            df_dense = st.session_state['keras_df']
            nth = st.session_state['nth']
        except:
            df_dense = pd.DataFrame()
            nth = 1

        add_layer = st.radio("추가 층의 종류", options = ['Dense', 'Dropout'], captions = ['Dense 층을 추가합니다.', '일부 정보를 off 하는 Dropout 층을 추가합니다.'])
        if add_layer == 'Dense':
            layer_units = int(st.number_input('Dense층의 unit수를 정수로 입력합니다.', min_value = 1, value = 8))
            layer_activation = st.selectbox("층의 활성화함수를 선택합니다.", ['relu', 'selu', 'elu', 'linear', 'leaky_relu'])

            if st.button("층 추가"):
                add_df = pd.DataFrame()
                add_df.loc[nth, 'Type'] = 'Dense'
                add_df.loc[nth, 'Units / Dropout'] = layer_units
                add_df.loc[nth, 'Activation'] = layer_activation
                df_dense = pd.concat([df_dense, add_df], axis = 0)
                st.session_state['keras_df'] = df_dense
                nth += 1
                st.session_state['nth'] = nth
        elif add_layer == 'Dropout':
            layer_dropout = test_size = st.slider("Dropout 비율 입력", 0.01, 0.99, 0.2)
            if st.button("층 추가"):
                add_df = pd.DataFrame()
                add_df.loc[nth, 'Type'] = 'Dropout'
                add_df.loc[nth, 'Units / Dropout'] = layer_dropout
                df_dense = pd.concat([df_dense, add_df], axis = 0)
                st.session_state['keras_df'] = df_dense
                nth += 1
                st.session_state['nth'] = nth

        df_dense2 = df_dense.copy()

        final_layer = pd.DataFrame()
        final_layer.loc['Final', 'Type'] = 'Dense'
        final_layer.loc['Final', 'Units / Dropout'] = final_units
        final_layer.loc['Final', 'Activation'] = final_activation

        df_dense2 = pd.concat([df_dense2, final_layer], axis = 0)
        st.dataframe(df_dense2)

        st.divider()
        st.title("모델 컴파일")
        st.write("옵티마이저, 손실함수, 평가지표를 선택하여 모델을 컴파일합니다.")

        st.header("옵티마이저 선택")
        optimizer = st.radio("사용할 옵티마이저를 선택합니다.", ['Adam', 'RMSprop', 'Nadam', 'SGD'])
        lr = st.number_input('옵티마이저의 학습률을 입력합니다.', value = 0.0001, min_value = 0.0, step = 1e-10, format = '%f')
        if optimizer == 'SGD':
            nesterov = st.toggle('SGD의 네스테로프 최적화 방법을 사용합니다.', value = True)
        else:
            epsilon = st.number_input('옵티마이저의 epsilon 값을 입력합니다.', value = 1e-07, min_value = 0.0, step = 1e-10, format = '%f')
        
        optimizer_dict = {'Adam' : Adam, 'RMSprop' : RMSprop, 'Nadam' : Nadam, 'SGD' : SGD}
        if optimizer == 'SGD':
            optimizer_apply = SGD(learning_rate = lr, nesterov = nesterov)
        else:
            optimizer_apply = optimizer_dict[optimizer](learning_rate = lr, epsilon = epsilon)

        st.header("손실함수 선택")
        if ml_type == '분류':
            if final_activation == 'sigmoid':
                st.write("손실함수는 binary_crossentropy로 자동 선택됩니다.")
                loss = 'binary_crossentropy'
            else:
                st.write("손실함수는 sparse_categorical_crossentropy로 자동 선택됩니다.")
                loss = 'sparse_categorical_crossentropy'
        else:
            loss = st.radio("사용할 손실함수를 선택합니다.", ['mae', 'mse', 'mape', 'msle'])

        st.header("평가지표 선택")
        if ml_type == '분류':
            if final_activation == 'sigmoid':
                metrics = st.multiselect('평가지표를 선택합니다.', ['accuracy', 'binary_accuracy', 'binary_crossentropy'])
            else:
                metrics = st.multiselect('평가지표를 선택합니다.', ['accuracy', 'sparse_categorical_crossentropy'])
        else:
            metrics = st.multiselect('평가지표를 선택합니다.', ['mae', 'mse', 'mape', 'msle', 'RootMeanSquaredError'])

        st.header("모델 컴파일")
        try:
            dl_model = st.session_state["dl_model"]
            dl_ready = True
        except:
            dl_ready = False

        if st.button("컴파일 시작"):
            is_1st_dense = True
            dl_cols = df_dense2.index
            layers = []
            for i in dl_cols:
                layer_type = df_dense2.loc[i, 'Type']
                if layer_type == 'Dense':
                    if is_1st_dense:
                        tmp_layer = Dense(units = int(df_dense2.loc[i, 'Units / Dropout']), activation = df_dense2.loc[i, 'Activation'], input_shape = (x_train.shape[1],))
                        is_1st_dense = False
                    else:
                        tmp_layer = Dense(units = int(df_dense2.loc[i, 'Units / Dropout']), activation = df_dense2.loc[i, 'Activation'])
                elif layer_type == 'Dropout':
                        tmp_layer = Dropout(rate = df_dense2.loc[i, 'Units / Dropout'])
                
                layers.append(tmp_layer)

            dl_model = Sequential(layers = layers)
            dl_model.compile(optimizer = optimizer_apply, loss = loss, metrics = metrics)
            st.session_state["dl_model"] = dl_model

            dl_model.summary(print_fn = st.write)
            dl_ready = True

            st.write("컴파일 완료")

        if dl_ready:
            st.divider()
            st.title("콜백 함수 정의")
            st.write("EarlyStopping 등의 콜백 함수를 정의합니다.")
            callbacks = []
            metric_li = ['loss', 'val_loss']
            for i in metrics:
                metric_li.append(i)
                metric_li.append("val_" + i)

            st.header("EarlyStopping 정의")
            if st.checkbox("EarlyStopping 사용", value = True):
                es_metric = st.selectbox("EarlyStopping 기준을 선택합니다.", metric_li)
                es_patience = int(st.number_input('EarlyStopping 실행 기준 Epoch 수를 입력합니다.', min_value = 1, value = 8))
                es_restore = st.toggle('EarlyStopping 발생 시 최적 모델의 가중치로 롤백할지 여부를 결정합니다.', value = True)
                callbacks.append(EarlyStopping(monitor = es_metric, patience = es_patience, restore_best_weights = es_restore))

            st.header('ReduceLROnPlateau 정의')
            if st.checkbox("ReduceLROnPlateau 사용", value = True):
                re_metric = st.selectbox("ReduceLROnPlateau 기준을 선택합니다.", metric_li)
                re_patience = int(st.number_input('ReduceLROnPlateau 실행 기준 Epoch 수를 입력합니다.', min_value = 1, value = 4))
                re_factor = st.number_input('ReduceLROnPlateau로 감소할 학습률의 배율을 입력합니다.', min_value = 0.0, max_value = 1.0, step = 0.0001, value = 0.1, format = '%f')
                re_min = st.number_input('감소할 학습률의 최소값을 입력합니다.', min_value = 0.0, max_value = 1.0, step = 0.000001, value = 0.0001, format = '%f')
                callbacks.append(ReduceLROnPlateau(monitor = re_metric, patience = re_patience, factor = re_factor, min_lr = re_min))

            st.divider()
            st.title("인공신경망 훈련")
            st.write("분할한 데이터와 구성한 신경망을 이용하여 인공신경망을 훈련합니다.")
            st.header("Epoch 횟수 입력")
            epochs = int(st.number_input('학습 반복 횟수를 입력합니다.', min_value = 1, value = 500))
            batch_size = int(st.number_input('배치 사이즈를 입력합니다.', min_value = 1, value = 32))

            try:
                dl_model_2 = st.session_state["dl_trained"]
                dl_done = True
            except:
                dl_done = False

            if st.button("훈련 시작"):
                with st.spinner("학습 진행중..."):
                    chk_time = dt.now().strftime("%Y%M%d_%H%M%S")
                    chk = ModelCheckpoint(f"./dl_model/{chk_time}.keras", save_best_only = True)
                    callbacks.append(chk)
                    st.session_state["chk"] = chk
                    history = dl_model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = epochs, batch_size = batch_size, verbose = 0, callbacks = callbacks)
                    st.success("모델 학습 완료!")
                    dl_done = True
                    st.session_state["dl_trained"] = dl_model
                    st.session_state['history'] = history
                    dl_model_2 = dl_model
            
            if dl_done:
                st.divider()
                st.title("훈련 결과 확인")
                st.header("학습 곡선")
                trans_dict = {'RootMeanSquaredError' : 'root_mean_squared_error'}
                result_li = ['loss']
                metrics_li = []
                for i in metrics:
                    if i in trans_dict.keys():
                        metrics_li.append(trans_dict[i])
                    else:
                        metrics_li.append(i)

                history = st.session_state['history']
                st.pyplot(yjk_dl_history_plot(history, metrics = metrics_li, show_results = False, for_st = True, fig_ratio = 3), use_container_width = False)

                result_li += metrics
                ev_train = dl_model_2.evaluate(x_train, y_train)
                ev_test = dl_model_2.evaluate(x_test, y_test)

                ev_df = pd.DataFrame()
                for i, v in enumerate(result_li):
                    ev_df.loc["Train", v] = ev_train[i]
                    ev_df.loc["Test", v] = ev_test[i]
                
                st.dataframe(ev_df)

                st.header("모델 다운로드")

                chkp = st.session_state['chk']
                with open(chkp.filepath, 'rb') as file:
                    st.download_button(
                            label="딥러닝 모델 다운로드",
                            data=file,
                            file_name='dl_model.keras'
                        )

            

                    

