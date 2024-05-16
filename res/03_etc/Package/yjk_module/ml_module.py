import pandas as pd
import numpy as np
import time, datetime, os
from copy import deepcopy as dc
from yjk_module.preprocessing import *
from sklearn.metrics import *
from tabulate import tabulate
from sklearn.model_selection import ParameterGrid, GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
from concurrent import futures
from prophet import Prophet
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor, BaggingClassifier, BaggingRegressor
from pycallgraphix.wrapper import register_method
import streamlit as st

class YjkQuantileRegressor():
    def fit(self, model_reg, model_cls, data, yname):
        if type(yname) != str:
            data = pd.concat([data, yname], axis = 1).copy()
            try:
                yname = yname.name
            except:
                yname = yname.columns[0]

        Q1 = data[yname].quantile(.25)
        Q2 = data[yname].median()
        Q3 = data[yname].quantile(.75)

        data['quantile_for_regression'] = data[yname].apply(lambda x : 1 if x < Q1 else(2 if x < Q2 else(3 if x < Q3 else 4)))
        self.model_cls = dc(model_cls)
        X_cls = data.drop([yname, 'quantile_for_regression'], axis = 1)
        Y_cls = data['quantile_for_regression']
        self.model_cls.fit(X_cls, Y_cls)

        df1 = data[data['quantile_for_regression'] == 1]
        df2 = data[data['quantile_for_regression'] == 2]
        df3 = data[data['quantile_for_regression'] == 3]
        df4 = data[data['quantile_for_regression'] == 4]

        X1 = df1.drop(['quantile_for_regression', yname], axis = 1)
        Y1 = df1[yname]
        X2 = df2.drop(['quantile_for_regression', yname], axis = 1)
        Y2 = df2[yname]
        X3 = df3.drop(['quantile_for_regression', yname], axis = 1)
        Y3 = df3[yname]
        X4 = df4.drop(['quantile_for_regression', yname], axis = 1)
        Y4 = df4[yname]

        self.model_reg_dict = {}
        self.model_reg_dict[1] = dc(model_reg)
        self.model_reg_dict[2] = dc(model_reg)
        self.model_reg_dict[3] = dc(model_reg)
        self.model_reg_dict[4] = dc(model_reg)

        self.model_reg_dict[1].fit(X1, Y1)
        self.model_reg_dict[2].fit(X2, Y2)
        self.model_reg_dict[3].fit(X3, Y3)
        self.model_reg_dict[4].fit(X4, Y4)

    def predict(self, data, soft = True):
        if soft:
            cls_arr = self.model_cls.predict_proba(data)
        else:
            cls_arr = pd.get_dummies(self.model_cls.predict(data))
        arr1 = pd.Series(self.model_reg_dict[1].predict(data))
        arr2 = pd.Series(self.model_reg_dict[2].predict(data))
        arr3 = pd.Series(self.model_reg_dict[3].predict(data))
        arr4 = pd.Series(self.model_reg_dict[4].predict(data))
        arr = pd.concat([arr1, arr2, arr3, arr4], axis = 1)

        result = (arr * cls_arr).sum(axis = 1).to_numpy()
        return result
    
    def score(self, x, y, scoring = 'r2', soft = True):
        x_test = x.copy()
        if type(y) == str:
            y_test = x_test.pop(y)
        else:
            y_test = y.copy()

        scoring_li = ['r2', 'mae', 'rmse', 'mse']
        scoring_dict = {'r2' : r2_score, 'mae' : mean_absolute_error, 'mse' : mean_squared_error, 'rmse' : mean_squared_error}
        y_pred = self.predict(x_test, soft = soft)

        if scoring not in scoring_li:
            raise Exception(f"입력한 scoring은 적합한 평가 기준이 아닙니다. {scoring_li} 중에서 적합한 값을 입력하세요.")

        score = scoring_dict[scoring](y_test, y_pred)
        if scoring == 'rmse':
            score = np.sqrt(score)
        
        return score

def yjk_confusion_matrix_metrics(data : np.array):
    if data.shape != (2, 2):
        raise Exception("데이터형이 혼동행렬이 아닙니다.")
    [[TN, FP], [FN, TP]] = data

    result = pd.DataFrame()
    result.loc['정확도', '값'] = (TN + TP) / (TN + FP + FN + TP)
    result.loc['정밀도', '값'] = (TP) / (FP + TP)
    result.loc['재현율', '값'] = (TP) / (FN + TP)
    result.loc['위양성율', '값'] = (FP) / (FP + TN)
    result.loc['특이성', '값'] = 1 - ((FP) / (FP + TN))

    tmp1 = (TP) / (FP + TP)
    tmp2 = (TP) / (FN + TP)
    result.loc['F1 Score'] = 2 * ((tmp1 * tmp2) / (tmp1 + tmp2))
    return result

def yjk_prophet_gridsearch(train, test, params, scoring = 'rmse', best_only = False, plot = False, figsize = (12, 4)):
    li_param = ParameterGrid(params)

    def tmp_func(train, test, i, scoring):

        m = Prophet(**i)
        m.fit(train)

        forecast = m.predict(test)

        pred = forecast[['ds', 'yhat']]
        pred_sort = pred.sort_values('ds')
        test_sort = test.sort_values('ds')

        if scoring == 'rmse':
            score = np.sqrt(mean_squared_error(test_sort['y'].values, pred_sort['yhat'].values))
        elif scoring == 'mae':
            score = mean_absolute_error(test_sort['y'].values, pred_sort['yhat'].values)
        else:
            score = r2_score(test_sort['y'].values, pred_sort['yhat'].values)
        
        return m, score, i
    
    result = []
    processes = []

    with futures.ThreadPoolExecutor() as executor:
        for p in li_param:
            processes.append(executor.submit(tmp_func, train, test, p, scoring))
        
        for p in futures.as_completed(processes):
            m, score, i = p.result()
            result.append({
                "model" : m,
                "params" : i,
                "score" : score
            })
    
    results = pd.DataFrame(result).sort_values("score").reset_index(drop = True)

    if plot:
        best_model = results['model'][0]

        future = best_model.make_future_dataframe(periods = len(test) + 7, freq = 'D')
        forecast = best_model.predict(future)

        fig = best_model.plot(forecast, figsize = figsize)
        fig.set_dpi(100)
        plt.show()

    if best_only:
        return results['model'][0], results['params'][0], results['score'][0]
    else:
        return results

class YjkRegressorSupport():
    def __init__(self, alert = True):
        self.model_name = None

        base_col = ['파라미터명', '파라미터 노트' , '파라미터 범위']
        self.model_dict = {}

        # -------------------- 회귀 모델들 ------------------------------
        self.model_dict['LinearRegressor'] = pd.DataFrame([
            [None, 'LinearRegressor는 특별한 하이퍼파라미터가 없음', None]
        ], columns = base_col).set_index('파라미터명')

        self.model_dict['Ridge'] = pd.DataFrame([
            ['alpha', '규제 강도, 클수록 규제가 강해지고 과소적합을 유도, 값은 10^n 형태로 사용', [0.001, 0.01, 0.1, 1, 10, 100]]
        ], columns = base_col).set_index('파라미터명')

        self.model_dict['Lasso'] = pd.DataFrame([
            ['alpha', '규제 강도, 클수록 규제가 강해지고 과소적합을 유도, 값은 10^n 형태로 사용', [0.001, 0.01, 0.1, 1, 10, 100]]
        ], columns = base_col).set_index('파라미터명')

        self.model_dict['SGDRegressor'] = pd.DataFrame([
            ['loss', '손실함수 지정, 기본은 squared_error', ['huber', 'squared_error', 'epsilon_insensitive', 'squared_epsilon_insensitive']],
            ['penalty', '규제의 종류를 지정, 기본값은 l2', [None, 'l1', 'l2', 'elasticnet']],
            ['alpha', '규제의 강도를 지정, 범위는 0 ~ 무한대, 로그스케일로 지정', [0.0001, 0.001, 0.01, 0.1, 1, 10]],
            ['max_iter', '수행할 최대 에포크 횟수, 기본 1000', [100, 300, 500, 1000]],
            ['tol', '성능향상이 안될 경우 학습을 종료하는 민감도 값, 기본 1e-3', [1e-4, 1e-3, 1e-2]],
        ], columns = base_col).set_index('파라미터명')

        self.model_dict['KneighborsRegressor'] = pd.DataFrame([
            ['n_neighbors', '데이터로부터 뽑는 최근접 이웃의 수, 정수값, 기본은 5', [3, 4, 5, 6, 7]],
            ['weights', '가중치 함수, 기본은 uniform(사용 안함), 콜백함수를 넣을수도 있음', ['uniform', 'distance']],
            ['p', 'float 혹은 int, 1은 맨하탄 거리 측정, 2는 유클리디안 거리 측정(기본값)', [1, 2]]
        ], columns = base_col).set_index('파라미터명')

        self.model_dict['LinearSVR'] = pd.DataFrame([
            ['loss', '손실함수 지정, 기본은 squared_hinge', ['epsilon_insensitive', 'squared_epsilon_insensitive']],
            ['C', '정규화의 강도의 역수, 작을수록 모델이 강하게 정규화됨', [0.01, 0.1, 1, 10]],
            ['max_iter', '에포크 횟수, 기본 1000', [100, 1000, 10000]],
            ['dual', '최적화 문제의 이중형식 또는 원시형식 선택', [True, False]]
        ], columns = base_col).set_index('파라미터명')

        self.model_dict['SVR'] = pd.DataFrame([
            ['C', '오류항에 대한 패널티, 값이 클수록 오류에 대한 패넡티가 커짐', [0.1, 1, 10]],
            ['kernel', '변환의 종류를 지정, 데이터를 높은 차원으로 매핑하는 함수, 기본 rbf', ['linear', 'poly', 'rbf', 'sigmoid']],
            ['degree', 'kernel이 poly일 경우 다항식의 차수, 기본 3', [2, 3, 4, 5]],
            # ['gamma', 'kernel이 rbf, poly, sigmoid일 경우 커널 계수 결정 방법, 기본 scale', ['auto', 'scale']],
            # ['coef0', 'kernel이 poly, sigmoid일 경우 독립항을 결정, 기본 0', [-10, 0.01, 0, 10]],
            # ['shrinking', '최적화를 위한 축소 휴리스틱 적용 여부', [True, False]],
            # ['max_iter', '에포크 횟수', [-1, 1000, 3000]],
        ], columns = base_col).set_index('파라미터명')
        
        self.model_dict['DecisionTreeRegressor'] = pd.DataFrame([
            ['criterion', '노드 분할의 기준', ['friedman_mse', 'squared_error', 'absolute_error', 'poisson']],
            ['splitter', '각 노드에서 분할을 선택하는 방식, best는 최선의 분할을 찾으며 random은 무작위', ['best', 'random']],
            ['max_depth', '나무의 깊이', [None, 3, 5, 7, 10, 20, 30]],
            ['min_samples_split', '노드를 분할하기 위한 최소 샘플 수, 갚이 클수록 분할이 적게 일어나 모델이 단순화', [2, 5, 10]],
            ['min_samples_leaf', '리프 노드가 되기 위한 최소 샘플 수, 값이 클수록 모델이 단순하고 과적합이 방지됨', [1, 2, 4]],
            ['max_features', '최적의 분할을 찾기 위해 고려할 최대 특성 수', ['sqrt', 'log2', None]]
        ], columns = base_col).set_index('파라미터명')

        self.model_dict['RandomForestRegressor'] = pd.DataFrame([
            ['criterion', '노드 분할의 기준', ['friedman_mse', 'squared_error', 'absolute_error', 'poisson']],
            ['max_depth', '나무의 깊이', [None, 3, 5, 7, 10, 20, 30]],
            ['min_samples_split', '노드를 분할하기 위한 최소 샘플 수, 갚이 클수록 분할이 적게 일어나 모델이 단순화', [2, 5, 10]],
            ['min_samples_leaf', '리프 노드가 되기 위한 최소 샘플 수, 값이 클수록 모델이 단순하고 과적합이 방지됨', [1, 2, 4]],
            ['max_features', '최적의 분할을 찾기 위해 고려할 최대 특성 수', ['sqrt', 'log2', None]]
        ], columns = base_col).set_index('파라미터명')

        self.model_dict['AdaBoostRegressor'] = pd.DataFrame([
            ['n_estimators', '학습기의 최대 개수', [50, 100, 300]],
            ['learning_rate', '학습률', [1, 0.1, 0.01, 0.001]],
            ['loss', '손실함수', ['linear', 'square', 'exponential']],
        ], columns = base_col).set_index('파라미터명')

        self.model_dict['GradientBoostingRegressor'] = pd.DataFrame([
            ['loss', '손실 함수', ['squared_error' , 'absolute_error', 'huber', 'quantile']],
            ['n_estimators', '학습기의 최대 개수', [100, 300]],
            ['subsample', '훈련 샘플 비율', [0.1, 0.5, 1]],
            ['min_samples_split', '노드 분할을 위한 최소 샘플 수', [2, 5, 10]],
            ['min_samples_leaf', '리프 노드가 되기 위한 최소 샘플 수', [1, 2, 5]],
            ['max_depth', 'Tree의 깊이', [3, 5, 7, 10, 20, None]],
            ['max_features', '최적 분할을 위해 고려하는 최대 특성 수', [None, 'sqrt', 'log2']]
        ], columns = base_col).set_index('파라미터명')

        self.model_dict['XGBRegressor'] = pd.DataFrame([
            ['n_estimators', '학습을 진행할 부스팅 라운드 수', [100, 300]],
            ['learning_rate', '학습률, 높을수록 속도가 빠르지만 과적합 가능성이 높음', [0.3, 0.5, 0.08, 0.1]],
            ['min_child_weight', '트리를 분할하는데 필요한 모든 관측치의 최소 가중치 합', [1, 0.8, 0.5]],
            ['max_depth', '트리 깊이', [6, 7, 9, 11, 0]],
            ['subsample', '트리를 구성하는 데이터 샘플링 비율', [0.5, 0.75, 1]],
            ['colsample_bytree', '트리를 구성하는 피처의 샘플링 비율', [0.6, 0.8, 1]],
        ], columns = base_col).set_index('파라미터명')    

        self.model_dict['LGBMRegressor'] = pd.DataFrame([
            ['num_leaves', '트리당 최대 리프 수', [31, 20, 40]],
            ['learning_rate', '학습률, 높을수록 속도가 빠르지만 과적합 가능성이 높음', [0.01, 0.05, 0.1]],
            ['n_estimators', '학습을 진행할 부스팅 라운드 수', [100, 300, 500]],
            ['max_depth', '트리 깊이', [5, 7, 9, 11, -1]],
            ['subsample', '트리를 구성하는 데이터 샘플링 비율', [0.8, 0.9, 1]],
            ['colsample_bytree', '트리를 구성하는 피처의 샘플링 비율', [0.6, 0.8, 1]],
            ['min_child_samples', '결정 트리의 리프 노드가 되기 위해 필요한 샘플 수', [10, 20, 30]],
            ['min_child_weight', '트리를 분할하는데 필요한 모든 관측치의 최소 가중치 합', [0.001, 0.01, 0.1]]
        ], columns = base_col).set_index('파라미터명')  

        self.model_dict['CatBoostRegressor'] = pd.DataFrame([
            # ['iterations', 'Tree 개수', [100, 500]],
            # ['depth', 'Tree의 깊이', [None, 3, 5, 7, 9]],
            ['learning_rate', '학습률 지정, 기본 0.009', [0.001, 0.009, 0.01]],
            ['random_strength', '무작위성 트리 구조 선택 강도, 과적합 조정용', [4, 6, 8]],
            ['l2_leaf_reg', 'L2 규제항의 계수', [4, 6, 8]]
        ], columns = base_col).set_index('파라미터명')
        # -------------------- 회귀 모델들 ------------------------------

        self.regressor_li = list(self.model_dict.keys())
        if alert:
            print(f"사용 가능한 모델 : {self.regressor_li}")

    def call_model(self, model_name = None, param_show = True, return_model = True):
        if not model_name:
            raise Exception("모델명을 입력하세요.")
        self.model_name = model_name

        model_dict = {'LinearRegressor' : LinearRegression(n_jobs = -1), 'Ridge' : Ridge(), 'Lasso' : Lasso(),
                      'SGDRegressor' : SGDRegressor(random_state = 0, early_stopping = True), 'KneighborsRegressor' : KNeighborsRegressor(n_jobs = -1),
                      'SVR' : SVR(), 'LinearSVR' : LinearSVR(random_state = 0), 'DecisionTreeRegressor' : DecisionTreeRegressor(random_state = 0),
                      'RandomForestRegressor' : RandomForestRegressor(random_state = 0), 'GradientBoostingRegressor' : GradientBoostingRegressor(random_state = 0),
                      'AdaBoostRegressor' : AdaBoostRegressor(estimator = DecisionTreeRegressor(max_depth = 7, random_state = 0), random_state = 0),
                      'XGBRegressor' : XGBRegressor(random_state = 0), 'LGBMRegressor' : LGBMRegressor(verbose = -1, random_state = 0),
                      'CatBoostRegressor' : CatBoostRegressor(verbose = 0, random_seed = 0)
                      }

        if param_show:
            print(f"[{model_name}의 파라미터]")
            print(tabulate(self.model_dict[model_name], headers = 'keys', tablefmt = 'psql',
                                 showindex = True, numalign = "right"))
        
        if return_model:
            return dc(model_dict[model_name])


    def get_params(self, model = None, param_grid = False, ignore = None):
        if not model:
            if not self.model_name:
                raise Exception("call_model 메서드로 모델명을 먼저 지정하세요.")
        
        if ignore:
            if type(ignore) == str:
                ignore = [ignore]

        if model:
            calling = self.model_dict[model]
        else:
            calling = self.model_dict[self.model_name]

        param_dict = {}

        for i in calling.index:
            if ignore:
                if i in ignore:
                    continue
            param_dict[i] = calling.loc[i, '파라미터 범위']
        
        if param_grid:
            return ParameterGrid(param_dict)
        else:
            return param_dict
        

class YjkClassifierSupport():
    @register_method
    def __init__(self, alert = True):
        self.model_name = None

        base_col = ['파라미터명', '파라미터 노트' , '파라미터 범위']
        self.model_dict = {}

        # -------------------- 분류 모델들 ------------------------------
        self.model_dict['LogisticRegression'] = pd.DataFrame([
            ['penalty', '규제의 종류, 기본값은 l2', [None, 'l1', 'l2', 'elasticnet']],
            ['C', '규제의 정도(정규화 강도의 역수), 값이 작을수록 강한 정규화', [0.001, 0.01, 0.1, 1, 10, 100]],
            ['max_iter', '에포크 횟수, 기본 100',  [100, 500]]
        ], columns = base_col).set_index('파라미터명')

        self.model_dict['SGDClassifier'] = pd.DataFrame([
            ['loss', '손실함수 지정, 기본은 hinge, 예시 외 다수 존재', ['hinge', 'log_loss', 'huber', 'modified_huber']],
            ['penalty', '규제의 종류를 지정, 기본값은 l2', [None, 'l1', 'l2', 'elasticnet']],
            ['alpha', '규제의 강도를 지정, 범위는 0 ~ 무한대, 로그스케일로 지정', [0.0001, 0.001, 0.01, 0.1, 1, 10]],
            ['max_iter', '수행할 최대 에포크 횟수, 기본 1000', [1000, 2000, 3000]],
            ['tol', '성능향상이 안될 경우 학습을 종료하는 민감도 값, 기본 1e-3', [1e-4, 1e-3, 1e-2]],
            ['learning_rate', '학습률 스케쥴링 전략', ['optimal', 'constant', 'invscaling', 'adaptive']],
            ['eta0', '초기 학습률', [0.01, 0.1, 0.5]]
        ], columns = base_col).set_index('파라미터명')

        self.model_dict['KNeighborsClassifier'] = pd.DataFrame([
            ['n_neighbors', '데이터로부터 뽑는 최근접 이웃의 수, 정수값, 기본은 5', [3, 4, 5, 6, 7]],
            ['weights', '가중치 함수, 기본은 uniform(사용 안함), 콜백함수를 넣을수도 있음', ['uniform', 'distance']],
            ['p', 'float 혹은 int, 1은 맨하탄 거리 측정, 2는 유클리디안 거리 측정(기본값)', [1, 2]]
        ], columns = base_col).set_index('파라미터명')

        self.model_dict['LinearSVC'] = pd.DataFrame([
            ['penalty', '규제의 종류를 지정, 기본값은 l2', ['l1', 'l2']],
            ['loss', '손실함수 지정, 기본은 squared_hinge', ['hinge', 'squared_hinge']],
            ['C', '정규화의 강도의 역수, 작을수록 모델이 강하게 정규화됨', [0.01, 0.1, 1, 10]],
            ['max_iter', '에포크 횟수, 기본 1000', [100, 1000, 10000]],
            ['dual', '최적화 문제의 이중형식 또는 원시형식 선택', [True, False]]
        ], columns = base_col).set_index('파라미터명')

        self.model_dict['SVC'] = pd.DataFrame([
            ['C', '오류항에 대한 패널티, 값이 클수록 오류에 대한 패넡티가 커짐', [0.1, 1, 10]],
            ['kernel', '변환의 종류를 지정, 데이터를 높은 차원으로 매핑하는 함수, 기본 rbf', ['linear', 'poly', 'rbf', 'sigmoid']],
            ['degree', 'kernel이 poly일 경우 다항식의 차수, 기본 3', [2, 3, 4, 5]],
            # ['gamma', 'kernel이 rbf, poly, sigmoid일 경우 커널 계수 결정 방법, 기본 scale', ['auto', 'scale']],
            # ['coef0', 'kernel이 poly, sigmoid일 경우 독립항을 결정, 기본 0', [-10, 0.01, 0, 10]],
            # ['shrinking', '최적화를 위한 축소 휴리스틱 적용 여부', [True, False]],
            # ['max_iter', '에포크 횟수', [-1, 1000, 3000]],
        ], columns = base_col).set_index('파라미터명')

        self.model_dict['GaussianNB'] = pd.DataFrame([
            ['var_smoothing', '예측을 위한 분산의 일부를 최대 분산에 추가해서 계산하는데 사용되는 값, 과적합 방지용', [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]]
        ], columns = base_col).set_index('파라미터명')

        self.model_dict['DecisionTreeClassifier'] = pd.DataFrame([
            ['criterion', '노드 분할의 기준, gini는 지니 불순도, entropy는 정보 이득을 사용', ['gini', 'entropy']],
            ['splitter', '각 노드에서 분할을 선택하는 방식, best는 최선의 분할을 찾으며 random은 무작위', ['best', 'random']],
            ['max_depth', '나무의 깊이', [None, 3, 5, 7, 10, 20, 30]],
            ['min_samples_split', '노드를 분할하기 위한 최소 샘플 수, 갚이 클수록 분할이 적게 일어나 모델이 단순화', [2, 5, 10]],
            ['min_samples_leaf', '리프 노드가 되기 위한 최소 샘플 수, 값이 클수록 모델이 단순하고 과적합이 방지됨', [1, 2, 4]],
            ['max_features', '최적의 분할을 찾기 위해 고려할 최대 특성 수', ['sqrt', 'log2', None]]
        ], columns = base_col).set_index('파라미터명')

        self.model_dict['RandomForestClassifier'] = pd.DataFrame([
            ['n_estimators', 'Tree의 개수', [100, 300, 500]],
            ['criterion', '노드 분할의 기준, gini는 지니 불순도, entropy는 정보 이득을 사용', ['gini', 'entropy']],
            ['max_depth', '나무의 깊이', [None, 3, 5, 7, 10, 20, 30]],
            ['min_samples_split', '노드를 분할하기 위한 최소 샘플 수, 갚이 클수록 분할이 적게 일어나 모델이 단순화', [2, 5, 10]],
            ['min_samples_leaf', '리프 노드가 되기 위한 최소 샘플 수, 값이 클수록 모델이 단순하고 과적합이 방지됨', [1, 2, 4]],
            ['max_features', '최적의 분할을 찾기 위해 고려할 최대 특성 수', ['sqrt', 'log2', None]]
        ], columns = base_col).set_index('파라미터명')

        self.model_dict['AdaBoostClassifier'] = pd.DataFrame([
            ['n_estimators', '학습기의 최대 개수', [50, 100, 300]],
            ['learning_rate', '학습률', [1, 0.1, 0.01, 0.001]]
        ], columns = base_col).set_index('파라미터명')

        self.model_dict['GradientBoostingClassifier'] = pd.DataFrame([
            ['loss', '손실 함수', ['log_loss', 'exponential']],
            ['n_estimators', '학습기의 최대 개수', [100, 300]],
            ['subsample', '훈련 샘플 비율', [0.1, 0.5, 1]],
            ['min_samples_split', '노드 분할을 위한 최소 샘플 수', [2, 5, 10]],
            ['min_samples_leaf', '리프 노드가 되기 위한 최소 샘플 수', [1, 2, 5]],
            ['max_depth', 'Tree의 깊이', [3, 5, 7, 10, 20, None]],
            ['max_features', '최적 분할을 위해 고려하는 최대 특성 수', [None, 'sqrt', 'log2']]
        ], columns = base_col).set_index('파라미터명')

        self.model_dict['XGBClassifier'] = pd.DataFrame([
            ['n_estimators', '학습을 진행할 부스팅 라운드 수', [100, 300]],
            ['learning_rate', '학습률, 높을수록 속도가 빠르지만 과적합 가능성이 높음', [0.3, 0.5, 0.08, 0.1]],
            ['min_child_weight', '트리를 분할하는데 필요한 모든 관측치의 최소 가중치 합', [1, 0.8, 0.5]],
            ['max_depth', '트리 깊이', [6, 7, 9, 11, 0]],
            ['subsample', '트리를 구성하는 데이터 샘플링 비율', [0.5, 0.75, 1]],
            ['colsample_bytree', '트리를 구성하는 피처의 샘플링 비율', [0.6, 0.8, 1]],
        ], columns = base_col).set_index('파라미터명')      

        self.model_dict['LGBMClassifier'] = pd.DataFrame([
            ['num_leaves', '트리당 최대 리프 수', [31, 20, 40]],
            ['learning_rate', '학습률, 높을수록 속도가 빠르지만 과적합 가능성이 높음', [0.01, 0.05, 0.1]],
            ['n_estimators', '학습을 진행할 부스팅 라운드 수', [100, 300, 500]],
            ['max_depth', '트리 깊이', [5, 7, 9, 11, -1]],
            ['subsample', '트리를 구성하는 데이터 샘플링 비율', [0.8, 0.9, 1]],
            ['colsample_bytree', '트리를 구성하는 피처의 샘플링 비율', [0.6, 0.8, 1]],
            ['min_child_samples', '결정 트리의 리프 노드가 되기 위해 필요한 샘플 수', [10, 20, 30]],
            ['min_child_weight', '트리를 분할하는데 필요한 모든 관측치의 최소 가중치 합', [0.001, 0.01, 0.1]]
        ], columns = base_col).set_index('파라미터명')  

        self.model_dict['CatboostClassifier'] = pd.DataFrame([
            # ['iterations', 'Tree 개수', [100, 500]],
            # ['depth', 'Tree의 깊이', [None, 3, 5, 7, 9]],
            ['learning_rate', '학습률 지정, 기본 0.009', [0.001, 0.009, 0.01]],
            ['random_strength', '무작위성 트리 구조 선택 강도, 과적합 조정용', [4, 6, 8]],
            ['l2_leaf_reg', 'L2 규제항의 계수', [4, 6, 8]]
        ], columns = base_col).set_index('파라미터명')
        # -------------------- 분류 모델들 ------------------------------

        self.classifier_li = list(self.model_dict.keys())
        if alert:
            print(f"사용 가능한 모델 : {self.classifier_li}")

    @register_method
    def call_model(self, model_name = None, param_show = True, return_model = True, include_proba = False):
        if not model_name:
            raise Exception("모델명을 입력하세요.")
        self.model_name = model_name

        model_dict = {'LogisticRegression' : LogisticRegression(n_jobs = -1), 'SGDClassifier' : SGDClassifier(random_state = 0, n_jobs = -1, early_stopping = True),
                      'KNeighborsClassifier' : KNeighborsClassifier(n_jobs = -1),'LinearSVC' : LinearSVC(random_state = 0),
                      'SVC' : SVC(probability = include_proba), 'GaussianNB' : GaussianNB(), 'DecisionTreeClassifier' : DecisionTreeClassifier(random_state = 0),
                      'RandomForestClassifier' : RandomForestClassifier(random_state = 0), 'GradientBoostingClassifier' : GradientBoostingClassifier(random_state = 0),
                      'AdaBoostClassifier' : AdaBoostClassifier(estimator = DecisionTreeClassifier(max_depth = 7, random_state = 0), random_state = 0, algorithm = 'SAMME'),
                      'XGBClassifier' : XGBClassifier(random_state = 0), 'LGBMClassifier' : LGBMClassifier(verbose = -1, random_state = 0),
                      'CatboostClassifier' : CatBoostClassifier(random_seed = 0, verbose = 0)
                      }

        if param_show:
            print(f"[{model_name}의 파라미터]")
            print(tabulate(self.model_dict[model_name], headers = 'keys', tablefmt = 'psql',
                                 showindex = True, numalign = "right"))
        
        if return_model:
            return dc(model_dict[model_name])

    @register_method
    def get_params(self, model = None, param_grid = False, ignore = None):
        if not model:
            if not self.model_name:
                raise Exception("call_model 메서드로 모델명을 먼저 지정하세요.")
        
        if ignore:
            if type(ignore) == str:
                ignore = [ignore]

        if model:
            calling = self.model_dict[model]
        else:
            calling = self.model_dict[self.model_name]

        param_dict = {}

        for i in calling.index:
            if ignore:
                if i in ignore:
                    continue
            param_dict[i] = calling.loc[i, '파라미터 범위']
        
        if param_grid:
            return ParameterGrid(param_dict)
        else:
            return param_dict

@register_method      
def yjk_classifier_multi_gridsearch(support : YjkClassifierSupport, x_train : pd.DataFrame, y_train : pd.Series,
                                    x_test : pd.DataFrame = None, y_test : pd.Series = None,
                                    randomized_search : bool = True, randomized_iter : int = 25, cv : int = 5,
                                    primary_score : str = 'accuracy',
                                    secondary_score = None, sort_by : str = 'test', include_models = None, exclude_models = None,
                                    include_proba : bool = False, time_log : bool = True, param_override : dict = {}, for_st : bool = False):
    base_exclude = []

    if type(x_test) != 'NoneType' and type(y_test) != 'NoneType':
        is_test = True
    else:
        is_test = False
        x_test = x_train
        y_test = y_train

    result_li = []
    total_time = 0
    if include_models:
        if include_models == 'all':
            models_li = support.classifier_li
        else:
            if type(include_models) == str:
                include_models = [include_models]
            models_li = include_models
    else:
        models_li = []
        for i in support.classifier_li:
            if i not in base_exclude:
                models_li.append(i)

    if exclude_models:
        if type(exclude_models) == str:
            exclude_models = [exclude_models]
    else:
        exclude_models = []

    if max([y_train.nunique(), y_test.nunique()]) > 2:
        binary = False
    else:
        binary = True

    for i in models_li:
        if i in exclude_models:
            continue
        if time_log:
            start = time.time()
        tmp_dict = {}
        tmp_model = support.call_model(i, param_show = False, include_proba = include_proba)
        params = support.get_params()

        if i not in  ['CatboostClassifier']:
            params = support.get_params()
        else:
            params = {}

        if i in param_override.keys():
            params = param_override[i]

        if randomized_search:
            search = RandomizedSearchCV(tmp_model, param_distributions = params, cv = cv,
                                        scoring = primary_score, n_iter = randomized_iter, n_jobs = -1, pre_dispatch = int(os.cpu_count()/2))
        else:
            search = GridSearchCV(tmp_model, param_grid = params, cv = cv, scoring = primary_score, n_jobs = -1, pre_dispatch = int(os.cpu_count()/2))

        search.fit(x_train, y_train)

        best_model = search.best_estimator_
        best_param = search.best_params_

        tmp_dict['model_name'] = i
        tmp_dict['best_model'] = best_model
        tmp_dict['best_param'] = best_param
        tmp_dict[f"{primary_score}_train"] = yjk_classification_score(best_model, x_train, y_train, metrics = primary_score, binary = binary)
        if is_test:
            tmp_dict[f"{primary_score}_test"] = yjk_classification_score(best_model, x_test, y_test, metrics = primary_score, binary = binary)

        if secondary_score:
            if type(secondary_score) == str:
                tmp_dict[f"{secondary_score}_train"] = yjk_classification_score(best_model, x_train, y_train, metrics = secondary_score, binary = binary)
                if is_test:
                    tmp_dict[f"{secondary_score}_test"] = yjk_classification_score(best_model, x_test, y_test, metrics = secondary_score, binary = binary)
            if type(secondary_score) == list:
                for j in secondary_score:
                    tmp_dict[f"{j}_train"] = yjk_classification_score(best_model, x_train, y_train, metrics = j, binary = binary)
                    if is_test:
                        tmp_dict[f"{j}_test"] = yjk_classification_score(best_model, x_test, y_test, metrics = j, binary = binary)

        result_li.append(tmp_dict)
        if time_log:
            end = time.time()
            sec = end - start
            total_time += sec
            time_result = str(datetime.timedelta(seconds=sec)).split(".")
            if for_st:
                st.write(f"{i} 모델 소요 시간 : {time_result[0]}")
            else:
                print(f"{i} 모델 소요 시간 : {time_result[0]}")

    result_df = pd.DataFrame(result_li)

    if sort_by == 'test' and is_test:
        result_df.sort_values(f"{primary_score}_test", ascending = False, inplace = True)
        result_df.reset_index(drop = True, inplace = True)
    else:
        result_df.sort_values(f"{primary_score}_train", ascending = False, inplace = True)
        result_df.reset_index(drop = True, inplace = True)

    result_df.set_index('model_name', inplace = True)
    if time_log:
        time_result_total = str(datetime.timedelta(seconds=total_time)).split(".")[0]
        if for_st:
            st.write(f"총 소요 시간 : {time_result_total}")
        else:
            print(f"총 소요 시간 : {time_result_total}")
    return result_df

def yjk_regressor_multi_gridsearch(support : YjkRegressorSupport, x_train : pd.DataFrame, y_train : pd.Series,
                                   x_test : pd.DataFrame = None, y_test : pd.Series = None,
                                    randomized_search : bool = True, randomized_iter : int = 25, cv : int = 5, primary_score : str = 'rmse',
                                    secondary_score = None, sort_by : str = 'test', include_models = None, exclude_models = None,
                                    time_log : bool = True, param_override : dict = {}, for_st : bool = False):
    base_exclude = []
    name_change = {'mae' : 'neg_mean_absolute_error', 'mse' : 'neg_mean_squared_error', 'rmse' : 'neg_root_mean_squared_error'}

    try:
        score = name_change[primary_score]
    except:
        score = primary_score

    if type(x_test) != 'NoneType' and type(y_test) != 'NoneType':
        is_test = True
    else:
        is_test = False
        x_test = x_train
        y_test = y_train

    result_li = []
    total_time = 0
    if include_models:
        if include_models == 'all':
            models_li = support.regressor_li
        else:
            if type(include_models) == str:
                include_models = [include_models]
            models_li = include_models
    else:
        models_li = []
        for i in support.regressor_li:
            if i not in base_exclude:
                models_li.append(i)

    if exclude_models:
        if type(exclude_models) == str:
            exclude_models = [exclude_models]
    else:
        exclude_models = []

    for i in models_li:
        if i in exclude_models:
            continue
        if time_log:
            start = time.time()
        tmp_dict = {}
        tmp_model = support.call_model(i, param_show = False)
        if i not in  ['LinearRegressor', 'CatBoostRegressor']:
            params = support.get_params()
        else:
            params = {}

        if i in param_override.keys():
            params = param_override[i]

        if randomized_search:
            search = RandomizedSearchCV(tmp_model, param_distributions = params, cv = cv,
                                        scoring = score, n_iter = randomized_iter, n_jobs = -1, pre_dispatch = int(os.cpu_count()/2))
        else:
            search = GridSearchCV(tmp_model, param_grid = params, cv = cv, scoring = score, n_jobs = -1, pre_dispatch = int(os.cpu_count()/2))

        search.fit(x_train, y_train)

        best_model = search.best_estimator_
        best_param = search.best_params_

        tmp_dict['model_name'] = i
        tmp_dict['best_model'] = best_model
        tmp_dict['best_param'] = best_param
        tmp_dict[f"{primary_score}_train"] = yjk_regression_score(best_model, x_train, y_train, metrics = primary_score)
        if is_test:
            tmp_dict[f"{primary_score}_test"] = yjk_regression_score(best_model, x_test, y_test, metrics = primary_score)

        if secondary_score:
            if type(secondary_score) == str:
                tmp_dict[f"{secondary_score}_train"] = yjk_regression_score(best_model, x_train, y_train, metrics = secondary_score)
                if is_test:
                    tmp_dict[f"{secondary_score}_test"] = yjk_regression_score(best_model, x_test, y_test, metrics = secondary_score)
            if type(secondary_score) == list:
                for j in secondary_score:
                    tmp_dict[f"{j}_train"] = yjk_regression_score(best_model, x_train, y_train, metrics = j)
                    if is_test:
                        tmp_dict[f"{j}_test"] = yjk_regression_score(best_model, x_test, y_test, metrics = j)

        result_li.append(tmp_dict)
        if time_log:
            end = time.time()
            sec = end - start
            total_time += sec
            time_result = str(datetime.timedelta(seconds=sec)).split(".")
            if for_st:
                st.write(f"{i} 모델 소요 시간 : {time_result[0]}")
            else:
                print(f"{i} 모델 소요 시간 : {time_result[0]}")

    result_df = pd.DataFrame(result_li)

    higher_better = ['r2']
    if primary_score in higher_better:
        sorting = False
    else:
        sorting = True

    if sort_by == 'test' and is_test:
        result_df.sort_values(f"{primary_score}_test", ascending = sorting, inplace = True)
        result_df.reset_index(drop = True, inplace = True)
    else:
        result_df.sort_values(f"{primary_score}_train", ascending = sorting, inplace = True)
        result_df.reset_index(drop = True, inplace = True)

    result_df.set_index('model_name', inplace = True)
    if time_log:
        time_result_total = str(datetime.timedelta(seconds=total_time)).split(".")[0]
        if for_st:
            st.write(f"총 소요 시간 : {time_result_total}")
        else:
            print(f"총 소요 시간 : {time_result_total}")
    return result_df

@register_method
def yjk_classification_score(model, x, y, metrics = 'accuracy', binary = True):
    simple_dict = {'accuracy' : accuracy_score}
    proba_dict = {'roc_auc' : roc_auc_score}
    multi_dict = {'f1_score' : f1_score, 'recall' : recall_score, 'precision' : precision_score}

    all_li = list(simple_dict.keys()) + list(proba_dict.keys()) + list(multi_dict.keys())
    if metrics not in all_li:
        raise Exception(f'{metrics}는 적용 가능한 평가지표가 아닙니다. sklearn.metrics의 get_scorer_names를 참조하세요.')
    
    if metrics in simple_dict.keys():
        score = simple_dict[metrics](y, model.predict(x))
    elif metrics in proba_dict.keys():
        if hasattr(model, "predict_proba"):
            if metrics == 'roc_auc':
                if binary:
                    score = roc_auc_score(y, model.predict_proba(x)[:, 1])
                else:
                    score = roc_auc_score(y, model.predict_proba(x), multi_class = 'ovr')
        else:
            return np.nan
    elif metrics in multi_dict.keys():
        if binary:
            score = multi_dict[metrics](y, model.predict(x))
        else:
            score = multi_dict[metrics](y, model.predict(x), average = 'macro')
    else:
        score = None
    return score

def yjk_regression_score(model, x, y, metrics = 'r2'):
    name_change = {'mae' : 'neg_mean_absolute_error', 'mse' : 'neg_mean_squared_error', 'rmse' : 'neg_root_mean_squared_error'}
    try:
        metrics = name_change[metrics]
    except:
        pass

    simple_dict = {'r2' : r2_score, 'neg_mean_absolute_error' : mean_absolute_error, 'neg_mean_squared_error' : mean_squared_error}
    sqrt_dict = {'neg_root_mean_squared_error' : mean_squared_error}

    all_li = list(simple_dict.keys()) + list(sqrt_dict.keys())
    if metrics not in all_li:
        raise Exception(f'{metrics}는 적용 가능한 평가지표가 아닙니다. sklearn.metrics의 get_scorer_names를 참조하세요.')
    
    if metrics in simple_dict.keys():
        score = simple_dict[metrics](y, model.predict(x))
    elif metrics in sqrt_dict.keys():
        score = sqrt_dict[metrics](y, model.predict(x))
        score = np.sqrt(score)
    else:
        score = None
    return score

@register_method
def yjk_classification_param_plot(model, x_train : pd.DataFrame, y_train : pd.Series, param_name : str, param_list : list,
                                  x_test : pd.DataFrame = None, y_test : pd.Series = None, scoring : str = 'accuracy',
                                  figsize : tuple = (12, 6), dpi : int = 100, summary : bool = 'full', as_tabulate : bool = False, for_st : bool = False):

    try:
        test_unique = y_test.nunique()
    except:
        test_unique = 2
    
    if max([y_train.nunique(), test_unique]) > 2:
        binary = False
    else:
        binary = True

    is_test = False
    result_train = []
    if x_test is not None and y_test is not None:
        is_test = True
        result_test = []

    for v in param_list:
        tmp_param = {param_name : v}
        tmp_model = dc(model)
        tmp_model.set_params(**tmp_param)

        tmp_model.fit(x_train, y_train)
        train_score = yjk_classification_score(tmp_model, x_train, y_train, metrics = scoring, binary = binary)
        if is_test:
            test_score = yjk_classification_score(tmp_model, x_test, y_test, metrics = scoring, binary = binary)

        result_train.append(train_score)
        if is_test:
            result_test.append(test_score)

    fig = plt.figure(figsize = figsize, dpi = dpi)
    plt.plot(result_train, label = 'Train')
    if is_test:
        plt.plot(result_test, label = 'Test')
        plt.legend()

    plt.xticks(range(0, len(param_list)), [str(i) for i in param_list])
    plt.xlabel(param_name)
    plt.ylabel(f"{scoring} (higher is better)")
    plt.grid()
    if not for_st:
        plt.show()

    if summary:
        if summary == 'simple':
            if is_test:
                idx_min = np.argmin(result_test)
                idx_max = np.argmax(result_test)
                print(f"{scoring}의 최소값은 {param_name}이 {param_list[idx_min]}일때 {result_test[idx_min]}")
                print(f"{scoring}의 최대값은 {param_name}이 {param_list[idx_max]}일때 {result_test[idx_max]}")
            else:
                idx_min = np.argmin(result_train)
                idx_max = np.argmax(result_train)
                print(f"{scoring}의 최소값은 {param_name}이 {param_list[idx_min]}일때 {result_train[idx_min]}")
                print(f"{scoring}의 최대값은 {param_name}이 {param_list[idx_max]}일때 {result_train[idx_max]}")
        else:
            if is_test:
                idx_min = np.argmin(result_test)
                idx_max = np.argmax(result_test)
                summary_df = pd.concat([
                    pd.Series(result_train, index = param_list, name = '훈련 데이터'),
                    pd.Series(result_test, index = param_list, name = '검증 데이터'),
                ], axis = 1)
                summary_df['Min-Max'] = ' '
                summary_df.iloc[idx_min, 2] = '최소값'
                summary_df.iloc[idx_max, 2] = '최대값'
                summary_df.index.name = param_name
                if as_tabulate:
                    print(tabulate(summary_df, headers = 'keys', tablefmt = 'psql', showindex = True, numalign = "right"))
                else:
                    if for_st:
                        return fig, summary_df
                    else:
                        return summary_df
            else:
                idx_min = np.argmin(result_train)
                idx_max = np.argmax(result_train)
                summary_df = pd.DataFrame(
                    pd.Series(result_train, index = param_list, name = '훈련 데이터'))
                summary_df['Min-Max'] = ' '
                summary_df.iloc[idx_min, 1] = '최소값'
                summary_df.iloc[idx_max, 1] = '최대값'
                summary_df.index.name = param_name
                if as_tabulate:
                    print(tabulate(summary_df, headers = 'keys', tablefmt = 'psql', showindex = True, numalign = "right"))
                else:
                    if for_st:
                        return fig, summary_df
                    else:
                        return summary_df
                
def yjk_regression_param_plot(model, x_train : pd.DataFrame, y_train : pd.Series, param_name : str, param_list : list,
                                  x_test : pd.DataFrame = None, y_test : pd.Series = None, scoring : str = 'r2',
                                  figsize : tuple = (12, 6), dpi : int = 100, summary : bool = 'full', as_tabulate : bool = False, for_st : bool = False):
    higher_better = ['r2']

    is_test = False
    result_train = []
    if x_test is not None and y_test is not None:
        is_test = True
        result_test = []

    for v in param_list:
        tmp_param = {param_name : v}
        tmp_model = dc(model)
        tmp_model.set_params(**tmp_param)

        tmp_model.fit(x_train, y_train)
        train_score = yjk_regression_score(tmp_model, x_train, y_train, metrics = scoring)
        if is_test:
            test_score = yjk_regression_score(tmp_model, x_test, y_test, metrics = scoring)

        result_train.append(train_score)
        if is_test:
            result_test.append(test_score)

    fig = plt.figure(figsize = figsize, dpi = dpi)
    plt.plot(result_train, label = 'Train')
    if is_test:
        plt.plot(result_test, label = 'Test')
        plt.legend()

    plt.xticks(range(0, len(param_list)), [str(i) for i in param_list])
    if scoring in higher_better:
        plt.ylabel(f"{scoring} (higher is better)")
    else:
        plt.ylabel(f"{scoring} (lower is better)")
    plt.xlabel(param_name)
    plt.grid()
    if not for_st:
        plt.show()

    if summary:
        if summary == 'simple':
            if is_test:
                idx_min = np.argmin(result_test)
                idx_max = np.argmax(result_test)
                print(f"{scoring}의 최소값은 {param_name}이 {param_list[idx_min]}일때 {result_test[idx_min]}")
                print(f"{scoring}의 최대값은 {param_name}이 {param_list[idx_max]}일때 {result_test[idx_max]}")
            else:
                idx_min = np.argmin(result_train)
                idx_max = np.argmax(result_train)
                print(f"{scoring}의 최소값은 {param_name}이 {param_list[idx_min]}일때 {result_train[idx_min]}")
                print(f"{scoring}의 최대값은 {param_name}이 {param_list[idx_max]}일때 {result_train[idx_max]}")
        else:
            if is_test:
                idx_min = np.argmin(result_test)
                idx_max = np.argmax(result_test)
                summary_df = pd.concat([
                    pd.Series(result_train, index = param_list, name = '훈련 데이터'),
                    pd.Series(result_test, index = param_list, name = '검증 데이터'),
                ], axis = 1)
                summary_df['Min-Max'] = ' '
                summary_df.iloc[idx_min, 2] = '최소값'
                summary_df.iloc[idx_max, 2] = '최대값'
                summary_df.index.name = param_name
                if as_tabulate:
                    print(tabulate(summary_df, headers = 'keys', tablefmt = 'psql', showindex = True, numalign = "right"))
                else:
                    if for_st:
                        return fig, summary_df
                    else:
                        return summary_df
            else:
                idx_min = np.argmin(result_train)
                idx_max = np.argmax(result_train)
                summary_df = pd.DataFrame(
                    pd.Series(result_train, index = param_list, name = '훈련 데이터'))
                summary_df['Min-Max'] = ' '
                summary_df.iloc[idx_min, 1] = '최소값'
                summary_df.iloc[idx_max, 1] = '최대값'
                summary_df.index.name = param_name
                if as_tabulate:
                    print(tabulate(summary_df, headers = 'keys', tablefmt = 'psql', showindex = True, numalign = "right"))
                else:
                    if for_st:
                        return fig, summary_df
                    else:
                        return summary_df
    
def yjk_make_ensemble_models(result_df : pd.DataFrame, exclude_models : list = [], include_models : list = []):
    if len(include_models):
        result_df = result_df.loc[include_models]
    
    ensemble_li = []

    for i in result_df.index:
        if i in exclude_models:
            continue
        else:
            ensemble_li.append((i, result_df.loc[i, 'best_model']))
    
    return ensemble_li

def yjk_classification_vs_ensemble(result_df : pd.DataFrame, x_train : pd.DataFrame, y_train : pd.Series,
                                   x_test : pd.DataFrame = None, y_test : pd.Series = None,
                                   exclude_models : list = [], include_models : list = [],
                                   stk_estimator = RandomForestClassifier(random_state = 0), voting : str = 'hard', cv : int = 5, n_estimators : int = 50,
                                   primary_score : str = 'accuracy', secondary_score : list = []):

    is_test = False
    if x_test is not None and y_test is not None:
        is_test = True

    try:
        test_unique = y_test.nunique()
    except:
        test_unique = 2
    
    if max([y_train.nunique(), test_unique]) > 2:
        binary = False
    else:
        binary = True

    ensemble_li = yjk_make_ensemble_models(result_df, exclude_models, include_models)
    solo_best = dc(result_df.iloc[0, 0])

    stk = StackingClassifier(ensemble_li, stk_estimator, cv = cv, n_jobs = int(os.cpu_count()/2))
    vot = VotingClassifier(ensemble_li, voting = voting, n_jobs = int(os.cpu_count()/2))
    bag = BaggingClassifier(solo_best, n_estimators = n_estimators, n_jobs = int(os.cpu_count()/2))
    boost = AdaBoostClassifier(solo_best, n_estimators = n_estimators)

    stk.fit(x_train, y_train)
    vot.fit(x_train, y_train)
    bag.fit(x_train, y_train)
    boost.fit(x_train, y_train)
    solo_best.fit(x_train, y_train)

    results = pd.DataFrame()
    results.loc['Solo_Best', 'estimator'] = solo_best
    results.loc['Voting', 'estimator'] = vot
    results.loc['Stacking', 'estimator'] = stk
    results.loc['Bagging', 'estimator'] = bag
    results.loc['Boosting', 'estimator'] = boost

    results[f'{primary_score}_train'] = results['estimator'].apply(lambda x : yjk_classification_score(x, x_train, y_train, primary_score, binary))
    if is_test:
        results[f'{primary_score}_test'] = results['estimator'].apply(lambda x : yjk_classification_score(x, x_test, y_test, primary_score, binary))

    if secondary_score:
        if type(secondary_score) == str:
            secondary_score = [secondary_score]
        for j in secondary_score:
            results[f"{j}_train"] = results['estimator'].apply(lambda x : yjk_classification_score(x, x_train, y_train, j, binary))
            if is_test:
                results[f"{j}_test"] = results['estimator'].apply(lambda x : yjk_classification_score(x, x_test, y_test, j, binary))

    if is_test:
        results.sort_values(f'{primary_score}_test', ascending = False, inplace = True)
    else:
        results.sort_values(f'{primary_score}_train', ascending = False, inplace = True)
    return results

def yjk_regression_vs_ensemble(result_df : pd.DataFrame, x_train : pd.DataFrame, y_train : pd.Series,
                                   x_test : pd.DataFrame = None, y_test : pd.Series = None,
                                   exclude_models : list = [], include_models : list = [],
                                   stk_estimator = RandomForestRegressor(random_state = 0), cv : int = 5, n_estimators : int = 50,
                                   primary_score : str = 'rmse', secondary_score : list = []):
    
    higher_better = ['r2']
    if primary_score in higher_better:
        sorting = False
    else:
        sorting = True

    is_test = False
    if x_test is not None and y_test is not None:
        is_test = True

    ensemble_li = yjk_make_ensemble_models(result_df, exclude_models, include_models)
    solo_best = dc(result_df.iloc[0, 0])

    stk = StackingRegressor(ensemble_li, stk_estimator, cv = cv, n_jobs = int(os.cpu_count()/2))
    vot = VotingRegressor(ensemble_li, n_jobs = int(os.cpu_count()/2))
    bag = BaggingRegressor(solo_best, n_estimators = n_estimators, n_jobs = int(os.cpu_count()/2))
    boost = AdaBoostRegressor(solo_best, n_estimators = n_estimators)

    stk.fit(x_train, y_train)
    vot.fit(x_train, y_train)
    bag.fit(x_train, y_train)
    boost.fit(x_train, y_train)
    solo_best.fit(x_train, y_train)

    results = pd.DataFrame()
    results.loc['Solo_Best', 'estimator'] = solo_best
    results.loc['Voting', 'estimator'] = vot
    results.loc['Stacking', 'estimator'] = stk
    results.loc['Bagging', 'estimator'] = bag
    results.loc['Boosting', 'estimator'] = boost

    results[f'{primary_score}_train'] = results['estimator'].apply(lambda x : yjk_regression_score(x, x_train, y_train, primary_score))
    if is_test:
        results[f'{primary_score}_test'] = results['estimator'].apply(lambda x : yjk_regression_score(x, x_test, y_test, primary_score))

    if secondary_score:
        if type(secondary_score) == str:
            secondary_score = [secondary_score]
        for j in secondary_score:
            results[f"{j}_train"] = results['estimator'].apply(lambda x : yjk_regression_score(x, x_train, y_train, j))
            if is_test:
                results[f"{j}_test"] = results['estimator'].apply(lambda x : yjk_regression_score(x, x_test, y_test, j))

    if is_test:
        results.sort_values(f'{primary_score}_test', ascending = sorting, inplace = True)
    else:
        results.sort_values(f'{primary_score}_train', ascending = sorting, inplace = True)
    return results

