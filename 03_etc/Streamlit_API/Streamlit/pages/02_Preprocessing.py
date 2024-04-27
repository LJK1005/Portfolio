import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
import streamlit as st

is_origin = False
st.title("데이터 전처리")
st.markdown("지정한 종속변수는 모든 데이터 전처리에서 제외됩니다. 분류분석의 경우 선택에 따라 라벨링을 수행할 수 있습니다.")

try:
    origin = st.session_state['origin']
    yname = st.session_state['yname']
    is_origin = True
except:
    st.write("원본 데이터가 입력되지 않았습니다.")

if is_origin:
    st.header("원본 데이터")
    st.dataframe(origin)
    st.write(f"종속변수 컬럼 : {yname}")
    st.divider()

    st.title("불필요 데이터 삭제")
    df1 = origin.copy()

    st.header("열 삭제")
    st.write("제거할 열을 선택합니다.")
    cols = list(df1.columns)
    cols.remove(yname)
    del_cols = st.multiselect("열 선택", cols)
    df2 = df1.drop(del_cols, axis = 1)
    st.dataframe(df2)

    st.header("행 삭제")
    st.write("제거할 행을 선택합니다.")
    del_idx = st.multiselect("행을 인덱스명으로 선택", df1.index)
    df3 = df2.drop(del_idx, axis = 0)
    st.dataframe(df3)

    st.divider()
    st.title("데이터 형변환")
    st.write("데이터를 필요에 따라 연속형 변수나 명목형 변수로 변환합니다.")
    df4 = df3.copy()

    st.markdown("- 아래 버튼으로 작업을 되돌립니다.")
    if st.button("Reset", type = "primary"):
        df4 = df3.copy()
        st.session_state["df_trans"] = df4

    st.markdown("- 변환할 컬럼과 데이터 타입을 선택합니다.")
    cols2 = list(df4.columns)
    cols2.remove(yname)
    trans_col = st.selectbox("컬럼명 선택", cols2)
    trans_to = st.radio("변환할 데이터형 선택", ["정수형", "실수형", "명목형"])
    try:
        df5 = st.session_state["df_trans"]
    except:
        df5 = df4.copy()

    if st.button("변환!"):
        try:
            if trans_to == "정수형":
                df5[trans_col] = df5[trans_col].astype("int")
            if trans_to == "실수형":
                df5[trans_col] = df5[trans_col].astype("float")
            if trans_to == "명목형":
                df5[trans_col] = df5[trans_col].astype("object")
            
            st.write("데이터 변환 성공")
            st.session_state["df_trans"] = df5
            
        except:
            st.write("데이터 변환 실패, 컬럼의 데이터를 다시 확인하세요.")

    col1, col2 = st.columns(2)
    with col2:
        st.write("변환된 데이터 타입")
        st.write(pd.DataFrame(df5.dtypes, columns = ['Type']))

    with col1:
        st.write("원본 데이터 타입")
        st.write(pd.DataFrame(df3.dtypes, columns = ['Type']))

    df6 = st.session_state["df_trans"].copy()
    st.markdown("- 변환 결과")
    st.dataframe(df6)

    st.divider()
    st.title("이상치 처리")
    st.write("사분위수를 기반으로 한 연속형 변수의 이상치를 정제합니다. 이상치 정제는 모든 컬럼에 대하여 같은 기준으로 수행합니다.")
    col3, col4 = st.columns(2)
    with col4:
        outline_col = list(df6.select_dtypes(["float", "int"]).columns)
        try:
            outline_col.remove(yname)
        except:
            pass
        outline_upper = {}
        outline_lower = {}
        outline_count = []
        for i in outline_col:
            tmp_S = df6[i]
            tmp_dict = {}
            tmp_Q1 = tmp_S.quantile(.25)
            tmp_Q3 = tmp_S.quantile(.75)
            tmp_IQR = tmp_Q3 - tmp_Q1
            outline_lower[i] = tmp_Q1 - tmp_IQR * 1.5
            outline_upper[i] = tmp_Q3 + tmp_IQR * 1.5
            cnt_h = len(tmp_S[tmp_S > outline_upper[i]])
            cnt_l = len(tmp_S[tmp_S < outline_lower[i]])
            tmp_dict["Name"] = i
            tmp_dict["Count"] = cnt_h + cnt_l
            outline_count.append(tmp_dict)

        st.markdown("- 수치형 데이터에 대한 이상치 개수")
        st.dataframe(pd.DataFrame(outline_count))

    with col3:
        outline_method = st.radio("이상치 정제 방법 정의", ["처리하지 않음", "경계값 대체", "중앙값 대체", "평균값 대체", "결측치 대체"],
                captions = ["", "각 컬럼별로 이상치의 기준이 되는 경계값으로 대체합니다.", "", "", "대체된 결측치는 결측치 처리에 의하여 처리됩니다."])
        
    if outline_method == "처리하지 않음":
        pass
    elif outline_method == "경계값 대체":
        for i in outline_col:
            df6.loc[df6[i] < outline_lower[i], i] = outline_lower[i]
            df6.loc[df6[i] > outline_upper[i], i] = outline_upper[i]
    elif outline_method == "중앙값 대체":
        for i in outline_col:
            df6.loc[df6[i] < outline_lower[i], i] = df6[i].median()
            df6.loc[df6[i] > outline_upper[i], i] = df6[i].median()
    elif outline_method == "평균값 대체":
        for i in outline_col:
            df6.loc[df6[i] < outline_lower[i], i] = df6[i].mean()
            df6.loc[df6[i] > outline_upper[i], i] = df6[i].mean()
    else:
        for i in outline_col:
            df6.loc[df6[i] < outline_lower[i], i] = np.nan
            df6.loc[df6[i] > outline_upper[i], i] = np.nan

    st.markdown("- 이상치 정제 결과")
    st.dataframe(df6)

    st.divider()

    st.title("결측치 처리")
    st.write("결측치 처리 방법을 정의하고 결측치를 정제합니다. 명목형 데이터는 Interpolate 옵션을 제외하고 모두 최빈값으로 대체됩니다.")

    df7 = df6.copy()
    
    cols_n = list(df7.select_dtypes("number").columns)
    cols_o = list(df7.columns)

    try:
        cols_n.remove(yname)
    except:
        try:
            cols_o.remove(yname)
        except:
            pass

    for i in cols_n:
        try:
            cols_o.remove(i)
        except:
            continue

    col5, col6 = st.columns(2)
    with col5:
        null_method = st.radio("결측치 정제 방법 정의", ["처리하지 않음", "중앙값 대체", "평균값 대체", "Interpolate"],
        captions = ["결측치가 있으면 머신러닝을 수행할 수 없습니다.", "", "", "명목형 데이터는 인근값으로 대체됩니다."])

    with col6:
        st.write("결측치 개수")
        null_df = pd.DataFrame(df7.isnull().sum(), columns = ["Null"])
        st.dataframe(null_df[null_df['Null'] > 0])

    if null_method == "처리하지 않음":
        pass
    elif null_method == "중앙값 대체":
        for i in cols_n:
            df7[i] = df7[i].fillna(df7[i].median())
        for i in cols_o:
            df7[i] = df7[i].fillna(df7[i].mode()[0])
    elif null_method == "평균값 대체":
        for i in cols_n:
            df7[i] = df7[i].fillna(df7[i].mean())
        for i in cols_o:
            df7[i] = df7[i].fillna(df7[i].mode()[0])
    else:
        df7.interpolate(inplace = True)
        df7 = df7.fillna(method = 'bfill').fillna(method = 'ffill')

    df7 = df7.astype(df6.dtypes.to_dict())

    st.markdown("- 결측치 처리 결과")
    st.dataframe(df7)

    st.divider()
    
    df8 = df7.copy()
    col_n = list(df8.select_dtypes("number").columns)
    try:
        col_n.remove(yname)
    except:
        pass
    st.title("연속형 변수 처리")
    st.write("연속형 변수에 대한 스케일링 방법을 정의하고 전처리를 수행합니다.")
    scale_method = st.radio("스케일링 방법 정의",
                            ["처리하지 않음", "StandardScaler", "MinMaxScaler", "RobustScaler", "LogScale"],
                            captions = ["", "", "", "", "상용로그를 적용하여 변환합니다. 데이터 컬럼에 0 이하의 실수가 없는지 확인하십시오."])
    
    if scale_method == "StandardScaler":
        scaler = StandardScaler()
        df8[col_n] = scaler.fit_transform(df8[col_n])
    if scale_method == "MinMaxScaler":
        scaler = MinMaxScaler()
        df8[col_n] = scaler.fit_transform(df8[col_n])
    if scale_method == "RobustScaler":
        scaler = RobustScaler()
        df8[col_n] = scaler.fit_transform(df8[col_n])
    if scale_method == "LogScale":
        df8[col_n] = np.log10(df8[col_n])

    st.markdown("- 스케일링 적용 결과")
    st.dataframe(df8)

    st.divider()
    col_o = list(df8.select_dtypes(['category', 'object']).columns)
    try:
        col_o.remove(yname)
    except:
        pass

    st.title("명목형 변수 처리")
    st.write("명목형 변수에 대한 인코딩 방법을 정의하고 전처리를 수행합니다.")
    encode_method = st.radio("인코딩 방법 정의",
                             ["처리하지 않음", "OneHotEncoder(더미화)", "OrdinalEncoder(라벨링)"],
                             captions = ["수치형 외의 데이터가 포함되어 있을 경우 머신러닝을 수행할 수 없습니다.", "", ""])
    
    if encode_method == "OneHotEncoder(더미화)":
        df_oh = pd.get_dummies(df8[col_o], dtype = 'int')
        df9 = df8.drop(col_o, axis = 1)
        df9 = pd.concat([df9, df_oh], axis = 1)
    elif encode_method == "OrdinalEncoder(라벨링)":
        ord = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = np.nan, encoded_missing_value = np.nan)
        tmp_df_o = pd.DataFrame(ord.fit_transform(df8[col_o]), columns = ord.get_feature_names_out())
        df9 = df8.drop(col_o, axis = 1)
        df9 = pd.concat([df9, tmp_df_o], axis = 1)
    else:
        df9 = df8.copy()

    st.markdown("- 인코딩 적용 결과")
    st.dataframe(df9)

    st.session_state['df_preprocess'] = df9

    st.divider()

    ml_type = st.session_state['ml_type']
    if ml_type == "분류":
        st.title("종속변수 라벨링")
        y_label = st.checkbox("종속변수 라벨링 여부를 선택합니다.")
        if y_label:
            le = LabelEncoder()
            df9[yname] = le.fit_transform(df9[yname])

        st.markdown("- 작업 결과")
        st.dataframe(df9)

        st.divider()

    @st.cache_data
    def convert_df(df):
        return df.to_csv(index = False).encode('utf-8')

    csv = convert_df(df9)

    st.download_button(
        label="전처리 결과 CSV 파일로 다운로드",
        data=csv,
        file_name='df_preprocess.csv',
        mime='text/csv',
    )