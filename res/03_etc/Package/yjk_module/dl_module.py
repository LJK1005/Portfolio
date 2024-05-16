import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from yjk_module.ml_module import yjk_classification_score, yjk_regression_score
from tensorflow import keras
from keras import losses, metrics, optimizers
from sklearn.model_selection import ParameterGrid

def yjk_dl_history_plot(history : keras.callbacks.History, metrics : str|list = None, is_val : bool = True, show_results : bool = True, dpi : int = 100, fig_ratio : int = 6, for_st : bool = False):
    df = pd.DataFrame(history.history)
    if metrics == None:
        fig = plt.figure(figsize = (fig_ratio * 2, fig_ratio), dpi = dpi)
        plt.plot(df['loss'], label = 'train_loss')
        if is_val:
            plt.plot(df['val_loss'], label = 'val_loss')
        plt.legend()
        plt.grid()
        if for_st:
            return fig
        else:
            plt.show()

    else:
        if type(metrics) == str:
            metrics = [metrics]

        fig, ax = plt.subplots(len(metrics) + 1, 1, figsize = (fig_ratio * 1.5, (len(metrics) + 1) * fig_ratio), dpi = dpi)
        ax = ax.flatten()
        fig.subplots_adjust(hspace = 0.2, wspace = 0.2)

        ax[0].plot(df['loss'], label = 'train_loss')
        if is_val:
            ax[0].plot(df['val_loss'], label = 'val_loss')
        ax[0].set_title('loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('loss')
        ax[0].legend()
        ax[0].grid()

        for i, v in enumerate(metrics):
            l = i + 1
            ax[l].plot(df[v], label = 'train_' + v)
            if is_val:
                ax[l].plot(df['val_' + v], label = 'val_' + v)
            ax[l].set_title(v)
            ax[l].set_xlabel('Epoch')
            ax[l].set_ylabel(v)
            ax[l].legend()
            ax[l].grid()
        if for_st:
            return fig
        else:
            plt.show()
    if show_results:
        tmp_df = pd.DataFrame()
        for i in history.history.keys():
            tmp_df.loc[i, 'Value'] = history.history[i][-1]
        
        print(tabulate(tmp_df, headers = 'keys', tablefmt = 'psql', showindex = True, numalign = "right"))


class YjkKerasHelper():
    def __init__(self, method : str = 'r', show_loss : bool = True, show_metrics : bool = True, show_optimizers : bool = True, no_show : bool = False):
        self.strategy = False
        if method == 'r':
            cols_loss = ['손실함수', '내용']
            df_loss = pd.DataFrame([
                ['mae', '예측값과 실제값의 절대 오차'],
                ['mape', '예측값과 실제값의 오차 비율'],
                ['mse', '예측값과 실제값의 제곱 오차'],
                ['msle', '예측값과 실제값의 로그값 차이'],
            ], columns = cols_loss).set_index('손실함수', drop = True)

            cols_metrics = ['평가지표', '내용']
            df_metrics = pd.DataFrame([
                ['mae', '예측값과 실제값의 절대 오차'],
                ['mape', '예측값과 실제값의 오차 비율'],
                ['mse', '예측값과 실제값의 제곱 오차'],
                ['msle', '예측값과 실제값의 로그값 차이'],
                ['rmse', 'MSE의 제곱근'],
            ], columns = cols_metrics).set_index('평가지표', drop = True)

            self.loss_dict = {'mae' : 'mae', 'mape' : 'mape', 'mse' : 'mse', 'msle' : 'msle'}
            self.metrics_dict = {'mae' : 'mae', 'mape' : 'mape', 'mse' : 'mse', 'msle' : 'msle', 'rmse' : 'RootMeanSquaredError'}

            if not no_show:
                if show_loss:
                    print('[손실함수]')
                    print(tabulate(df_loss, headers = 'keys', tablefmt = 'psql', showindex = True, numalign = "right"), end = "\n\n")
                if show_metrics:    
                    print('[평가지표]')
                    print(tabulate(df_metrics, headers = 'keys', tablefmt = 'psql', showindex = True, numalign = "right"), end = "\n\n")

        else:
            cols_loss = ['손실함수', '내용', '비고']
            df_loss = pd.DataFrame([
                ['binary_crossentropy', '예측값과 실제값간 교차 엔트로피 손실', '이진 분류에 사용, 라벨링 필요'],
                ['sparse_categorical_crossentropy', '예측값과 실제값간 교차 엔트로피 손실', '다중 클래스 분류에 사용, 라벨링 필요'],
                ['categorical_crossentropy', '클래스 확률 분포와 실제 값 간의 교차 엔트로피 손실', '다중 클래스 분류에 사용, 원핫인코딩 필요'],
            ], columns = cols_loss).set_index('손실함수', drop = True)

            cols_metrics = ['평가지표', '내용', '비고']
            df_metrics = pd.DataFrame([
                ['accuracy', '모델의 예측값과 실제값이 일치하는 비율', '모든 분류에 사용가능'],
                ['binary_accuracy', '모델의 예측값과 실제값이 일치하는 비율', '이진 분류에 사용'],
                ['binary_crossentropy', '예측값과 실제값간 교차 엔트로피 손실', '이진 분류에 사용, 라벨링 필요'],
                ['sparse_categorical_crossentropy', '예측값과 실제값간 교차 엔트로피 손실', '다중 클래스 분류에 사용, 라벨링 필요'],
                ['categorical_crossentropy', '클래스 확률 분포와 실제 값 간의 교차 엔트로피 손실', '다중 클래스 분류에 사용, 원핫인코딩 필요'],
            ], columns = cols_metrics).set_index('평가지표', drop = True)

            self.loss_dict = {'binary_crossentropy' : 'binary_crossentropy', 'sparse_categorical_crossentropy' : 'sparse_categorical_crossentropy', 'categorical_crossentropy' : 'categorical_crossentropy'}
            self.metrics_dict = {'binary_crossentropy' : 'binary_crossentropy', 'sparse_categorical_crossentropy' : 'sparse_categorical_crossentropy', 'categorical_crossentropy' : 'categorical_crossentropy',
                                 'accuracy' : 'accuracy', 'binary_accuracy' : 'binary_accuracy'}

            if not no_show:
                if show_loss:
                    print('[손실함수]')
                    print(tabulate(df_loss, headers = 'keys', tablefmt = 'psql', showindex = True, numalign = "right"), end = "\n\n")
                if show_metrics:    
                    print('[평가지표]')
                    print(tabulate(df_metrics, headers = 'keys', tablefmt = 'psql', showindex = True, numalign = "right"), end = "\n\n")
                
        optimizer_li = ['adam', 'adagrad', 'nadam', 'rmsprop', 'sgd']
        if not no_show:
            if show_optimizers:
                print(f"사용 가능한 옵티마이저 : {optimizer_li}")
    
    def set_strategy(self, loss : str, optimizer : str = 'adam', metrics : str|list = None):
        metrics_trans = {'rmse' : 'root_mean_squared_error'}
        if type(metrics) == str:
            metrics = [metrics]

        self.optimizer = optimizer
        self.loss = self.loss_dict[loss]
        self.metrics = []
        if metrics:
            for i in metrics:
                self.metrics.append(self.metrics_dict[i])
            self.metrics_for_plot = []
            for i in metrics:
                if i in metrics_trans.keys():
                    self.metrics_for_plot.append(metrics_trans[i])
                else:
                    self.metrics_for_plot.append(i)
        self.strategy = True

    def compile(self, model : keras.models.Sequential, optimizer_lr : float = 0.0001, optimizer_epsilon_nesterov : float|bool = None):
        optimizer_dict = {
            'adam' : optimizers.Adam,
            'adagrad' : optimizers.Adagrad,
            'nadam' : optimizers.Nadam,
            'rmsprop' : optimizers.RMSprop,
            'sgd' : optimizers.SGD
        }
        
        params = {}
        params['learning_rate'] = optimizer_lr
        if type(optimizer_epsilon_nesterov) == bool:
            params['nesterov'] = optimizer_epsilon_nesterov
        elif type(optimizer_epsilon_nesterov) == float:
            params['epsilon'] = optimizer_epsilon_nesterov
        optimizer = optimizer_dict[self.optimizer](**params)
        
        try:
            model.compile(optimizer = optimizer, loss = self.loss, metrics = self.metrics)
            print('컴파일 완료')
        except:
            print('컴파일 실패, 파라미터를 다시 점검하세요.')
    
def yjk_dl_gridsearch(func, param_grid : list|dict, higher : bool = True):
    if type(param_grid) == dict:
        param_grid = list(ParameterGrid(param_grid))
    best_model = None
    best_param = None
    best_score = None
    best_history = None

    for i in param_grid:
        score, model, history = func(**i)
        if best_score == None:
            best_score = score
            best_param = i
            best_model = model
            best_history = history
        if higher == score > best_score:
            best_score = score
            best_param = i
            best_model = model
            best_history = history
        
    print(f"최적 파라미터 : {best_param}")
    print(f"최적 점수 : {best_score}")

    return best_model, best_history