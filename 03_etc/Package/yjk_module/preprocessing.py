import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tabulate import tabulate

class YjkPreprocessor():
    """머신러닝에 필요한 데이터 전처리를 수행하는 객체입니다.
    IQR을 기준으로 하는 이상치 정제, 결측치 처리, 스케일링, 명목형 변수에 대한 인코딩을 지원합니다.
    객체 생성 후 set_strategy로 각 프로세스에 적용할 방법을 정의할 수 있습니다.
    fit_transform 메서드를 사용하여 데이터를 전처리를 수행할 수 있습니다. (set_strategy 메서드로 적용 방법을 정의하지 않으면 사용할 수 없습니다.)
    이후 transform 메서드로 동일한 유형의 데이터를 입력하면 fit_transform과 동일한 기준으로 전처리를 수행합니다.
    """
    def __init__(self):
        self.check_outliner = False
        self.check_null = False
        self.check_scaler = False
        self.check_encoder = False
        self.check_fit = False

    def set_strategy(self, outline = 'q', null = 'median', scaler = 's', encoder = 'o', alert = True):
        """데이터 전처리 방법을 정의합니다. 이 메서드를 실행해야 fit_transform 메서드를 사용할 수 있습니다.

        Args:
            outline (str, optional): 이상치 정제 방법을 정의합니다 'q'는 이상치 경계값으로 대체합니다. 'm'은 중앙값으로 대체합니다. 'a'는 평균값으로 대체합니다. 'd'는 이상치를 삭제합니다. 'n'은 결측치로 대체합니다. 결측치 대체 시 이후 결측치 처리 프로세스에 의해 처리됩니다. None은 이상치를 처리하지 않도록 합니다. Defaults to 'q'.
            null (str, optional): 결측치 처리 방법을 정의합니다. 'median'은 중앙값으로 대체합니다. 'mean'은 평균값으로 대체합니다. 'time'은 시계열에 따른 보간값으로 대체합니다. 'del'은 결측치를 삭제합니다. 명목형 데이터의 경우 'time'과 'median'은 최빈값으로 대체하며 'time'은 인근값으로 대체됩니다. None은 결측치를 처리하지 않도록 합니다. Defaults to 'median'.
            scaler (str, optional): 사용할 스케일러를 정의합니다. 's'는 StandardScaler를 사용합니다. 'm'은 MinMaxScaler를 사용합니다. 'r'은 RobustScaler를 사용합니다. None은 스케일링을 수행하지 않습니다. Defaults to 's'.
            encoder (str, optional): 범주형 데이터에 대한 인코더 사용을 정의합니다. 'o'는 원핫인코딩을 수행합니다. 'l'은 레이블인코딩을 수행합니다. 레이블인코딩은 사용을 권장하지 않습니다. 'ord'는 오디널인코딩을 수행합니다. transform 메서드에서 미확인 관측치는 결측치로 처리됩니다. None은 인코딩을 수행하지 않습니다. Defaults to 'o'.
        """
        outline_li = ['q', 'Q', 'm', 'M', 'n', 'N', None, 'a', 'A', 'd', 'D']
        null_li = ['median', 'mean', 'time', None, 'd', 'D']
        scaler_li = ['s', 'S', 'r', 'R', 'm', 'M', None]
        encoder_li = ['o', 'O', 'l', 'L', None, 'ord']

        outline_dict = {'q' : '경계값 대체', 'm' : '중앙값 대체', 'n' : '결측치 대체', None : '처리하지 않음', 'a' : '평균값 대체', 'd' : '이상치 삭제'}
        null_dict = {'median' : '중앙값 대체', 'mean' : '평균값 대체', 'time' : '시계열 처리', None : '처리하지 않음', 'd' : '결측치 삭제'}
        scaler_dict = {'s' : 'StandardScaler', 'm' : 'MinMaxScaler', 'r' : 'RobustScaler', None : '처리하지 않음'}
        encoder_dict = {'o' : '원핫인코딩', 'l' : '레이블인코딩', 'ord' : '오디널인코딩', None : '처리하지 않음'}
        
        if self.check_fit:
            print("이미 Fit된 데이터가 있습니다. 객체에 데이터를 다시 입력하세요")
            self.datetime = None
            self.check_fit = False

        if outline not in outline_li:
            raise Exception("이상치의 올바른 처리방법을 입력하세요. 'q', 'm', 'n', 'a', 'd', None 값만 허용됩니다.")

        try:
            self.outliner_strategy = outline.lower()
        except:
            self.outliner_strategy = outline
        self.check_outliner = True

        if null not in null_li:
            raise Exception("결측치의 올바른 처리방법을 입력하세요. 'median', 'mean', 'time', 'd', None 값만 허용됩니다.")
        
        if null == 'time':
            print("시계열 데이터의 결측치 처리방법을 입력했습니다. 데이터프레임에 시계열데이터가 있는지 확인하십시오.")
        
        self.null_strategy = null
        self.check_null = True

        if scaler not in scaler_li:
            raise Exception("스케일러의 올바른 방법을 입력하세요. 's', 'm', 'r', None 값만 허용됩니다.")
        
        try:
            self.scaler_strategy = scaler.lower()
        except:
            self.scaler_strategy = scaler
        self.check_scaler = True

        if encoder not in encoder_li:
            raise Exception("인코더의 올바른 방법을 입력하세요. 'o', 'l', 'ord', None 값만 허용됩니다.")

        try:
            self.encoder_strategy = encoder.lower()
        except:
            self.encoder_strategy = encoder
        self.check_encoder = True

        strategy_summary = pd.DataFrame([
            ['이상치 처리', outline_dict[self.outliner_strategy]],
            ['결측치 처리', null_dict[self.null_strategy]],
            ['스케일러', scaler_dict[self.scaler_strategy]],
            ['인코딩', encoder_dict[self.encoder_strategy]]
        ], columns = ['범주', '처리방법'])

        if alert:
            print("아래와 같이 처리합니다.")
            print(tabulate(strategy_summary, headers = 'keys', tablefmt = 'psql', showindex = False, numalign = "right"))

    def fit_transform(self, data = pd.DataFrame(), datetime = None, dt_trans = False, yname = None, ylabeling = False, drop = None, to_category = None, alert = True):
        """전처리의 기준이 되는 데이터를 입력하고 전처리 결과를 반환받는 메서드입니다. set_strategy 메서드를 먼저 사용해야 사용할 수 있습니다.

        Args:
            df (DataFrame): 전처리에 사용할 데이터프레임을 입력합니다.
            datetime (str or bool, optional): 정렬 기준이 되는 시계열 데이터 컬럼명을 입력합니다. set_strategy에서 null을 time으로 했을 경우 필수 항목입니다. 해당 시계열 데이터 컬럼의 타입은 반드시 datetime64 이어야 합니다. Defaults to None.
            yname(str or bool, optional) : 종속변수 컬럼이 데이터에 포함되어 있을 경우 해당 컬럼명을 입력합니다. 입력된 컬럼명은 처리를 진행하지 않습니다. Defaults to None.
            ylabeling(str or bool, optional) : 종속변수 컬럼이 범주형 데이터일 경우 레이블 인코딩 여부를 설정할 수 있습니다. Defaults to False.
            drop(str or bool, optional) : 원핫인코더의 첫번째 열 삭제 여부를 결정합니다. None, 'first', 'if_binary'중 하나를 입력합니다. Defaults to None.
            to_category(str or list, optional) : 데이터에서 카테고리 타입으로 변경하고 싶은 컬럼명을 문자열이나 리스트로 입력합니다. Defaults to None.
            alert(bool, optional) : 데이터프레임 정보 출력 여부를 설정합니다. Defaults to True.

        Returns:
            DataFrame: 전처리 결과를 데이터프레임으로 반환합니다.
        """

        # 기본 판별변수
        self.datetime = datetime
        self.dt_trans = dt_trans
        self.yname = None
        self.yalbeling = False
        self.to_category = None
        Y = None

        # 필수 입력사항 체크 및 예외처리
        check_li = []
        if not self.check_outliner:
            check_li.append("이상치 처리")
        if not self.check_null:
            check_li.append("결측치 처리")
        if not self.check_scaler:
            check_li.append("스케일링")
        if not self.check_encoder:
            check_li.append("인코딩")
        if len(check_li) > 0:
            raise Exception(f"{', '.join(check_li)}에 대한 방법이 정의되지 않았습니다.")
        
        if self.null_strategy == 'time':
            if datetime == None:
                raise Exception("시계열 데이터 컬럼명이 입력되지 않았습니다.")
            if datetime not in data.columns:
                raise Exception("입력한 컬럼명이 데이터프레임에 없습니다.")
            if data[datetime].isnull().sum() > 0:
                raise Exception("시계열 데이터에 결측치가 있습니다.")

        # 데이터 복사 및 기본 정보 추출
        df = data.copy()
        self.col_ord = list(df.columns)
        cols = df.columns

        # 입력받은 카테고리 타입 열 목록 변환
        if to_category:
            self.to_category = to_category
            if type(to_category) == str:
                self.to_category = [to_category]
            df[to_category] = df[to_category].astype('category')

        # 데이터타입별 열 목록 추출
        cols_n = list(df.select_dtypes(include = 'number').columns)
        cols_o = list(df.select_dtypes(['object', 'category']).columns)
        cols_other = list(cols.copy())

        for i in cols_n:
            cols_other.remove(i)
        for i in cols_o:
            cols_other.remove(i)

        # 종속변수 처리
        if yname:
            self.yname = yname
            if self.yname in cols_n:
                cols_n.remove(self.yname)
            if self.yname in cols_o:
                cols_o.remove(self.yname)
            if self.yname in cols_other:
                cols_other.remove(self.yname)
            Y = df[self.yname]
            if ylabeling: # 종속변수 레이블인코딩
                self.yalbeling = True
                self.ylabel = LabelEncoder()
                Y = pd.Series(self.ylabel.fit_transform(Y), name = Y.name, index = Y.index)

        # 데이터프레임 정보 출력
        if alert:
            print(f"데이터프레임 크기 : {df.shape}")

            null_df = df.isnull().sum()
            null_df = null_df[null_df > 0]

            if len(null_df):
                print("\n[열별 결측치 개수]")
                null_df = pd.DataFrame(np.array([null_df.index, null_df.values]).T,
                                            columns = ["열명", "결측치 수"])
                print(tabulate(null_df, headers = 'keys', tablefmt = 'psql', showindex = False,
                            numalign = "right"), end = "\n\n")
            else:
                print("데이터프레임에 결측치는 없습니다.")

            if len(cols_n):
                print(f"연속형 데이터 컬럼 : {list(cols_n)}, 총 {len(cols_n)}개")
            
            if len(cols_o):
                print(f"명목형 데이터 컬럼 : {list(cols_o)}, 총 {len(cols_o)}개")
                
            if len(cols_other):
                print(f"기타 데이터 컬럼 : {cols_other}, 총 {len(cols_other)}개")
            if datetime:
                print(f"시계열 데이터 컬럼명 : {datetime}")
            if yname:
                print(f"종속변수 컬럼명 : {yname} / 종속변수 라벨링 여부 : {self.yalbeling}")

        # 시계열 데이터 처리
        if datetime:
            df.sort_values(datetime, ascending = False, inplace = True)
            if dt_trans:
                self.dt_trans = True
                df[datetime] = pd.to_numeric(df[datetime]) / (10 ** 18)

        # 이상치 처리
        if self.outliner_strategy:
            if len(cols_n):
                Q1_dict = {}
                Q3_dict = {}
                IQR_dict = {}
                self.lower_dict = {}
                self.upper_dict = {}
                self.outline_dict = {}
            
                for i, v in enumerate(cols_n):
                    Q1_dict[v] = df[v].quantile(.25)
                    Q3_dict[v] = df[v].quantile(.75)
                    IQR_dict[v] = Q3_dict[v] - Q1_dict[v]
                    self.upper_dict[v] = Q3_dict[v] + 1.5 * IQR_dict[v]
                    self.lower_dict[v] = Q1_dict[v] - 1.5 * IQR_dict[v]

                    if self.outliner_strategy == 'q':
                        df.loc[df[v] < self.lower_dict[v], v] = self.lower_dict[v]
                        df.loc[df[v] > self.upper_dict[v], v] = self.upper_dict[v]
                        
                    elif self.outliner_strategy == 'a':
                        self.outline_dict[v] = df[v].mean()
                        df.loc[df[v] < self.lower_dict[v], v] = self.outline_dict[v]
                        df.loc[df[v] > self.upper_dict[v], v] = self.outline_dict[v]
                        
                    elif self.outliner_strategy == 'm':
                        self.outline_dict[v] = df[v].median()
                        df.loc[df[v] < self.lower_dict[v], v] = self.outline_dict[v]
                        df.loc[df[v] > self.upper_dict[v], v] = self.outline_dict[v]

                    elif self.outliner_strategy == 'n':
                        df.loc[df[v] < self.lower_dict[v], v] = np.nan
                        df.loc[df[v] > self.upper_dict[v], v] = np.nan

                    else:
                        df.drop(df[df[v] < self.lower_dict[v]].index, axis = 0, inplace = True)
                        df.drop(df[df[v] > self.upper_dict[v]].index, axis = 0, inplace = True)

        # 결측치 처리
        if self.null_strategy:
            if self.null_strategy != 'time':
                self.mode_dict_o = {}
                self.mode_dict_other = {}
                self.median_dict = {}
                self.mean_dict = {}

                if len(cols_o):
                    for i in cols_o:
                        self.mode_dict_o[i] = df[i].mode()[0]
                if len(cols_other):
                    for i in cols_other:
                        self.mode_dict_other[i] = df[i].mode()[0]

                if self.null_strategy == 'median':
                    if len(cols_n):
                        for i, v in enumerate(cols_n):
                            self.median_dict[v] = df[v].median()
                            df[v] = df[v].fillna(self.median_dict[v])
                    if len(cols_o):
                        for i, v in enumerate(cols_o):
                            df[v] = df[v].fillna(self.mode_dict_o[v])
                    if len(cols_other):
                        for i, v in enumerate(cols_other):
                            df[v] = df[v].fillna(self.mode_dict_other[v])

                elif self.null_strategy == 'mean':
                    if len(cols_n):
                        for i, v in enumerate(cols_n):
                            self.mean_dict[v] = df[v].mean()
                            df[v] = df[v].fillna(self.mean_dict[v])
                    if len(cols_o):
                        for i, v in enumerate(cols_o):
                            df[v] = df[v].fillna(self.mode_dict_o[v])
                    if len(cols_other):
                        for i, v in enumerate(cols_other):
                            df[v] = df[v].fillna(self.mode_dict_other[v])
                
                else:
                    df.dropna(inplace = True)

            else:
                if len(cols_n):
                    for i, v in enumerate(cols_n):
                        df[v] = df[v].interpolate()
                        df[v] = df[v].fillna(method = 'bfill')
                        df[v] = df[v].fillna(method = 'ffill')
                if len(cols_o):
                    for i, v in enumerate(cols_o):
                        df[v] = df[v].fillna(method = 'bfill')
                        df[v] = df[v].fillna(method = 'ffill')
                if len(cols_other):
                    for i, v in enumerate(cols_other):
                        df[v] = df[v].fillna(method = 'bfill')
                        df[v] = df[v].fillna(method = 'ffill')

        # 데이터 컬럼 동기화         
        idx = df.index

        if self.yname:
            Y = Y.loc[idx]            
                        
        if self.scaler_strategy or self.encoder_strategy:
            df_n = df[cols_n].copy()
            df_o = df[cols_o].copy()
            df_other = df[cols_other].copy()

            # 스케일링
            if self.scaler_strategy:
                if len(cols_n):
                    if self.scaler_strategy == 's':
                        self.scaler = StandardScaler()
                    elif self.scaler_strategy == 'm':
                        self.scaler = MinMaxScaler()
                    else:
                        self.scaler = RobustScaler()

                    df_n = pd.DataFrame(self.scaler.fit_transform(df_n), columns = cols_n)

            # 인코딩
            if self.encoder_strategy:
                if len(cols_o):
                    if self.encoder_strategy == 'l':
                        self.le_dict = {}
                        for i in cols_o:
                            self.le_dict[i] = LabelEncoder()
                            df_o[i] = self.le_dict[i].fit_transform(df_o[i])
                    elif self.encoder_strategy == 'o':
                        try:
                            self.oh = OneHotEncoder(sparse = False, handle_unknown = 'ignore', drop = drop)
                        except:
                            self.oh = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore', drop = drop)
                        tmp_df_o = pd.DataFrame(self.oh.fit_transform(df_o), columns = self.oh.get_feature_names_out())
                        df_o = tmp_df_o.copy()
                    else:
                        self.ord = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = np.nan, encoded_missing_value = np.nan)
                        tmp_df_o = pd.DataFrame(self.ord.fit_transform(df_o), columns = self.ord.get_feature_names_out())
                        df_o = tmp_df_o.copy()

            # 데이터 결합 및 리턴
            if len(df_n):
                df_n.index = idx
            if len(df_o):
                df_o.index = idx

            if self.yname:
                df2 = pd.concat([df_n, df_o, df_other], axis = 1)
                df2.index = idx
                df2 = pd.concat([df2, Y], axis = 1)
                self.check_fit = True
                df2.sort_index(inplace = True)
            else:
                df2 = pd.concat([df_n, df_o, df_other], axis = 1)
                df2.index = idx
                self.check_fit = True
                df2.sort_index(inplace = True)

            return df2
        else:
            return df
    
    def transform(self, data = pd.DataFrame(), alert = True):
        """추가적인 데이터를 전처리할 때 사용하는 메서드입니다. fit_transform 메서드를 사용한 후 사용할 수 있습니다.
        입력하는 데이터의 컬럼 이름이나 데이터형이 fit_transform의 데이터 컬럼과 동일한지 확인하십시오.
        fit_transform 메서드에서 yname을 지정한 경우 해당 yname명에 해당하는 컬럼은 필수로 들어갈 필요는 없습니다.

        Args:
            data (DataFrame): 전처리를 수행할 데이터프레임입니다.
            alert (bool, optional) : 데이터프레임 정보 출력 여부를 설정합니다. Defaults to True.

        Returns:
            DataFrame: 전처리를 수행한 데이터프레임을 반환합니다.
        """
        if not self.check_fit:
            raise Exception("객체에 데이터가 Fit되지 않았습니다. fit_transform으로 데이터를 객체에 입력하세요.")

        # 데이터 복사 및 열 순서 정렬
        df2 = data.copy()
        if self.yname:
            try:
                df2 = df2[self.col_ord]
            except:
                col_ord_2 = self.col_ord.copy()
                col_ord_2.remove(self.yname)
                df2 = df2[col_ord_2]
        cols_2 = df2.columns
        Y = None

        # 데이터 카테고리 타입 변경
        if self.to_category:
            df2[self.to_category] = df2[self.to_category].astype('category')

        # 데이터 타입 추출
        cols_n_2 = list(df2.select_dtypes(include = 'number').columns)
        cols_o_2 = list(df2.select_dtypes(['object', 'category']).columns)
        cols_other_2 = list(cols_2.copy())

        for i in cols_n_2:
            cols_other_2.remove(i)
        for i in cols_o_2:
            cols_other_2.remove(i)

        # 종속변수 처리
        if self.yname:
            if self.yname in cols_n_2:
                cols_n_2.remove(self.yname)
            if self.yname in cols_o_2:
                cols_o_2.remove(self.yname)
            if self.yname in cols_other_2:
                cols_other_2.remove(self.yname)
            try:
                Y = df2[self.yname]
                Y = pd.Series(self.ylabel.transform(Y), index = Y.index, name = Y.name)
            except:
                pass

        cols_other_2 = np.array(cols_other_2)

        # 데이터프레임 정보 출력
        if alert:
            print(f"데이터프레임 크기 : {df2.shape}")

            null_df = df2.isnull().sum()
            null_df = null_df[null_df > 0]

            if len(null_df):
                print("\n[열별 결측치 개수]")
                null_df = pd.DataFrame(np.array([null_df.index, null_df.values]).T,
                                            columns = ["열명", "결측치 수"])
                print(tabulate(null_df, headers = 'keys', tablefmt = 'psql', showindex = False,
                            numalign = "right"), end = "\n\n")
            else:
                print("데이터프레임에 결측치는 없습니다.")

        # 시계열 데이터 처리
        if self.datetime:
            df2.sort_values(self.datetime, ascending = False, inplace = True)
            if self.dt_trans:
                df2[self.datetime] = pd.to_numeric(df2[self.datetime])

        # 이상치 처리
        if self.outliner_strategy:
            for i, v in enumerate(cols_n_2):
                if self.outliner_strategy == 'q':
                    df2.loc[df2[v] < self.lower_dict[v], v] = self.lower_dict[v]
                    df2.loc[df2[v] > self.upper_dict[v], v] = self.upper_dict[v]
                    
                elif self.outliner_strategy == 'a':
                    df2.loc[df2[v] < self.lower_dict[v], v] = self.outline_dict[v]
                    df2.loc[df2[v] > self.upper_dict[v], v] = self.outline_dict[v]
                    
                elif self.outliner_strategy == 'm':
                    df2.loc[df2[v] < self.lower_dict[v], v] = self.outline_dict[v]
                    df2.loc[df2[v] > self.upper_dict[v], v] = self.outline_dict[v]

                elif self.outliner_strategy == 'n':
                    df2.loc[df2[v] < self.lower_dict[v], v] = np.nan
                    df2.loc[df2[v] > self.upper_dict[v], v] = np.nan

                else:
                    df2.drop(df2[df2[v] < self.lower_dict[v]].index, axis = 0, inplace = True)
                    df2.drop(df2[df2[v] > self.upper_dict[v]].index, axis = 0, inplace = True)

        # 결측치 처리
        if self.null_strategy:
            if self.null_strategy != 'time':
                if self.null_strategy == 'median':
                    if len(cols_n_2):
                        for i, v in enumerate(cols_n_2):
                            df2[v] = df2[v].fillna(self.median_dict[v])
                    if len(cols_o_2):
                        for i, v in enumerate(cols_o_2):
                            df2[v] = df2[v].fillna(self.mode_dict_o[v])
                    if len(cols_other_2):
                        for i, v in enumerate(cols_other_2):
                            df2[v] = df2[v].fillna(self.mode_dict_other[v])

                elif self.null_strategy == 'mean':
                    if len(cols_n_2):
                        for i, v in enumerate(cols_n_2):
                            df2[v] = df2[v].fillna(self.mean_dict[v])
                    if len(cols_o_2):
                        for i, v in enumerate(cols_o_2):
                            df2[v] = df2[v].fillna(self.mode_dict_o[v])
                    if len(cols_other_2):
                        for i, v in enumerate(cols_other_2):
                            df2[v] = df2[v].fillna(self.mode_dict_other[v])

                else:
                    df2.dropna(inplace = True)
                    
            else:
                if len(cols_n_2):
                    for i, v in enumerate(cols_n_2):
                        df2[v] = df2[v].interpolate()
                        df2[v] = df2[v].fillna(method = 'bfill')
                        df2[v] = df2[v].fillna(method = 'ffill')
                if len(cols_o_2):
                    for i, v in enumerate(cols_o_2):
                        df2[v] = df2[v].fillna(method = 'bfill')
                        df2[v] = df2[v].fillna(method = 'ffill')
                if len(cols_other_2):
                    for i, v in enumerate(cols_other_2):
                        df2[v] = df2[v].fillna(method = 'bfill')
                        df2[v] = df2[v].fillna(method = 'ffill')

        # 데이터 동기화
        idx_2 = df2.index

        if self.yname:
            try:
                Y = Y.loc[idx_2]
            except:
                pass
                
        if self.scaler_strategy or self.encoder_strategy:
            df_n_2 = df2[cols_n_2].copy()
            df_o_2 = df2[cols_o_2].copy()
            df_other_2 = df2[cols_other_2].copy()
            
            # 스케일링
            if self.scaler_strategy:
                if len(cols_n_2):
                    df_n_2 = pd.DataFrame(self.scaler.transform(df_n_2), columns = cols_n_2)

            # 인코딩
            if self.encoder_strategy:
                if len(cols_o_2):
                    if self.encoder_strategy == 'l':
                        for i in cols_o_2:
                            df_o_2[i] = self.le_dict[i].transform(df_o_2[i])
                    elif self.encoder_strategy == 'o':
                        tmp_df_o_2 = pd.DataFrame(self.oh.transform(df_o_2), columns = self.oh.get_feature_names_out())
                        df_o_2 = tmp_df_o_2.copy()
                    else:
                        tmp_df_o_2 = pd.DataFrame(self.ord.transform(df_o_2), columns = self.ord.get_feature_names_out())
                        df_o_2 = tmp_df_o_2.copy()

            # 데이터 결합 및 리턴
            if len(df_n_2):
                df_n_2.index = idx_2
            if len(df_o_2):
                df_o_2.index = idx_2
            
            if self.yname and Y:
                df2_2 = pd.concat([df_n_2, df_o_2, df_other_2], axis = 1)
                df2_2.index = idx_2
                df2_2 = pd.concat([df2_2, Y], axis = 1)
                df2_2.sort_index(inplace = True)
            else:
                df2_2 = pd.concat([df_n_2, df_o_2, df_other_2], axis = 1)
                df2_2.index = idx_2
                df2_2.sort_index(inplace = True)


            return df2_2
        else:
            return df2

def yjk_ttsplit_combined(X, Y, test_size = 0.25, random_state = 0, stratify = None, alert = True):
    """sklearn의 train_test_split 함수를 사용하지만 독립변수와 종속변수를 결합한 값을 리턴받습니다.

    Args:
        X (DataFrame): 독립변수 혹은 전체 데이터를 입력합니다.
        Y (Series or str): 종속변수 데이터 혹은 컬럼명을 입력합니다.
        test_size (float, optional): test_size를 입력합니다. Defaults to 0.25.
        random_state (int, optional): random_state를 입력합니다. Defaults to 0.
        stratify (Bool or Series, optional): stratify를 입력합니다. Defaults to None.
        alert(Bool, optional) : 종속변수 컬럼명 출력 여부를 설정합니다. Defaults to True.

    Returns:
        DataFrame: 독립변수와 종속변수가 결합된 데이터를 반환합니다.
    """
    if type(Y) == pd.Series:
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = test_size, random_state = random_state, stratify = stratify)
        train = pd.concat([x_train, y_train], axis = 1)
        test = pd.concat([x_test, y_test], axis = 1)
        name = Y.name
    if type(Y) == str:
        name = Y
        X = X.copy()
        Y2 = X.pop(Y)
        if stratify:
            x_train, x_test, y_train, y_test = train_test_split(X, Y2, test_size = test_size, random_state = random_state, stratify = Y2)
        else:    
            x_train, x_test, y_train, y_test = train_test_split(X, Y2, test_size = test_size, random_state = random_state)
        train = pd.concat([x_train, y_train], axis = 1)
        test = pd.concat([x_test, y_test], axis = 1)
    
    if alert:
        print(f"종속변수 : {name}")
    return train, test

def yjk_quantile_labeling(df : pd.DataFrame = pd.DataFrame(), col : str = None, col_name : str = None, is_category : bool = True):
    """데이터프레임과 수치형 데이터를 가진 컬럼명을 입력받아 해당 컬럼의 사분위수를 기준으로 라벨링을 시행합니다.

    Args:
        df (pd.DataFrame): 원본 데이터프레임. Defaults to pd.DataFrame().
        col (str): 라벨링을 수행할 컬럼명. Defaults to None.
        col_name (str): 라벨링 결과를 담을 컬럼명. Defaults to None.
        is_category (bool, optional): True로 설정하면 라벨링 결과는 category 타입이 됩니다 False일 경우 int 타입으로 반환됩니다. Defaults to True.

    Returns:
        DataFrame: 라벨링 결과를 포함한 데이터프레임을 반환합니다.
    """
    
    if col not in df.select_dtypes('number').columns:
        raise Exception(f"{col}행은 수치형 데이터가 아닙니다.")
    
    if col_name == None:
        col_name = col + "_quantile"

    if col_name in df.columns:
        raise Exception(f"신규 컬럼명 {col_name}은(는) 원본 데이터프레임에 이미 존재합니다.")

    df2 = df.copy()

    Q1 = df2[col].quantile(.25)
    Q2 = df2[col].median()
    Q3 = df2[col].quantile(.75)

    df2[col_name] = 0
    
    df2[col_name] = df2[col].apply(lambda x : 0 if x < Q1 else(1 if x < Q2 else (2 if x < Q3 else 3)))

    if is_category:
        df2[col_name] = df2[col_name].astype('category')
    
    return df2

def yjk_show_vif(data : pd.DataFrame, sort : bool = True, round : int = 2):
    vif = pd.DataFrame()
    col_n = data.select_dtypes('number').columns
    data = data[col_n]

    vif['Feature'] = data.columns
    vif['VIF'] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    if sort:
        vif.sort_values('VIF', ascending = False, inplace = True)
        vif.reset_index(drop = True, inplace = True)
    return vif.round(round)

def yjk_filter_vif(data = pd.DataFrame, tol : float = 10.0, show_result : bool = True):
    data2 = data.copy()
    del_cols = []
    while True:
        tmp = yjk_show_vif(data2)
        if tmp.iloc[0, 1] < tol:
            break
        else:
            data2.drop(tmp.iloc[0, 0], axis = 1, inplace = True)
            del_cols.append(tmp.iloc[0, 0])
    
    if show_result:
        print(f"삭제된 열 : {del_cols}")
        print(tabulate(yjk_show_vif(data2).T, headers = 'keys', tablefmt = 'psql', showindex = True, numalign = "right"))

    return data2