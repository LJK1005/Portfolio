import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

plt.rcParams["font.family"] = 'Malgun Gothic'
plt.rcParams["font.size"] = 6
plt.rcParams["figure.figsize"] = (4, 3)
plt.rcParams["axes.unicode_minus"] = False

origin = pd.DataFrame()

st.title("YJK's ML Platform")

st.header("데이터 불러오기")
load_file_method = st.radio("데이터 불러오는 방법 선택",
                ["파일 직접 업로드", "URL에서 다운로드"],
                captions = ["csv와 xlsx 파일 지원", "URL 링크는 csv 혹은 xlsx 파일을 직접 지정해야 함"])

if load_file_method == "파일 직접 업로드":
    uploaded_file = st.file_uploader("파일 업로드")
    if uploaded_file is not None:
        try:
            origin = pd.read_csv(uploaded_file)
        except:
            try:
                origin = pd.read_excel(uploaded_file)
            except:
                origin = pd.DataFrame()
else:
    url = st.text_input("데이터 URL 입력", "URL")

    if url[-5:] == ".xlsx":
        origin = pd.read_excel(url)
    elif url[-4:] == ".csv":
        origin = pd.read_csv(url)
    else:
        origin = pd.DataFrame()

if len(origin):
    st.session_state['origin'] = origin
    st.divider()

    st.title("업로드 데이터 확인")
    try:
        st.dataframe(origin)
    except:
        st.write("아직 데이터가 업로드되지 않았습니다.")

    st.divider()

    st.header("종속변수 컬럼과 분석 타입 선택")
    yname = st.selectbox("종속변수 컬럼 선택", origin.columns)
    ml_type = st.radio("분석 타입 선택",
                    ["회귀", "분류"], captions = ["회귀분석을 실행합니다.", "분류분석을 실행합니다."])
    
    st.session_state['yname'] = yname
    st.session_state['ml_type'] = ml_type

    st.divider()

    st.title("EDA - 데이터 기본 정보")
    st.header("데이터 크기")
    st.write(f"\n{origin.shape}")
    st.divider()
    st.header("데이터프레임 info")

    buffer = io.StringIO()
    origin.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    X = origin.copy()
    Y = X.pop(yname)
    st.divider()
    st.header("결측치 개수")
    null_df = pd.DataFrame(origin.isnull().sum(), columns = ["Null"])
    st.dataframe(null_df)

    st.divider()
    st.header("독립변수 기초통계량 - 수치형 데이터")
    try:
        st.dataframe(X.describe())
    except:
        st.write("데이터에 수치형 데이터가 없음")
    st.divider()
    st.header("독립변수 기초통계량 - 범주형 데이터")
    try:
        st.dataframe(X.describe(include = "O"))
    except:
        st.write("데이터에 범주형 데이터가 없음")

    st.divider()

    if ml_type == "회귀":
        st.header("종속변수 기초통계량")
        st.dataframe(Y.describe())
    else:
        st.header("종속변수 데이터 개수")
        st.dataframe(Y.value_counts())
    
    st.divider()

    st.title("그래프 조회")

    st.header("수치형 변수에 대한 Boxplot")
    try:
        fig = plt.figure(figsize = (6, 3))
        sns.boxplot(data = origin)
        st.pyplot(fig, use_container_width = False, clear_figure = True)
    except:
        st.write("데이터에 수치형 데이터가 없음")

    st.divider()
    
    st.header("각 변수에 대한 그래프 조회")
    col_n = origin.select_dtypes("number").columns
    col_o = origin.select_dtypes(["object", "category"]).columns

    col_plot = st.selectbox("컬럼명 선택", origin.columns)

    if col_plot in col_n:
        fig, ax = plt.subplots(2, 1, figsize = (6, 6))
        fig.subplots_adjust(hspace = 0.3)
        sns.histplot(data = origin[col_plot], kde = True, ax = ax[0])
        sns.boxplot(data = origin[col_plot], orient = 'h', ax = ax[1])
        ax[0].set_title("Histogram")
        ax[1].set_title("Boxplot")
        st.pyplot(fig, use_container_width = False, clear_figure = True)
    elif col_plot in col_o:
        fig, ax = plt.subplots(2, 1, figsize = (6, 6))
        fig.subplots_adjust(hspace = 0.3)
        sns.countplot(data = origin[col_plot], ax = ax[0], color = 'green')
        ax[1].pie(origin[col_plot].value_counts(), labels = origin[col_plot].value_counts().index,
                  autopct='%0.1f%%', explode = np.ones(origin[col_plot].nunique())/50)
        ax[0].set_title("Countplot")
        ax[1].set_title("Pieplot")
        st.pyplot(fig, use_container_width = False, clear_figure = True)        
    st.divider()
    st.header("산점도 그래프")
    show_pariplot = st.checkbox("산점도 그래프 표시")
    if show_pariplot:
        if ml_type == '회귀':
            g = sns.pairplot(origin)
        else:
            g = sns.pairplot(origin, hue = yname)
        st.pyplot(g.fig, use_container_width = False, clear_figure = True)

    st.divider()
    st.header("종속변수를 포함하는 그래프 조회")
    col_plot_2 = st.selectbox("컬럼명 선택", X.columns)
    if col_plot_2 in col_n:
        if ml_type == "회귀":
            fig = plt.figure(figsize = (6, 3), dpi = 100)
            sns.regplot(data = origin, x = col_plot_2, y = yname, line_kws = {'color' : "#ff5500"})
            st.pyplot(fig, use_container_width = False, clear_figure = True)
        else:
            fig = plt.figure(figsize = (6, 3), dpi = 100)
            sns.histplot(data = origin, x = col_plot_2, hue = yname, kde = True, fill = True)
            st.pyplot(fig, use_container_width = False, clear_figure = True)
    elif col_plot_2 in col_o:
        if ml_type == "회귀":
            fig = plt.figure(figsize = (6, 3), dpi = 100)
            sns.histplot(data = origin, x = yname, hue = col_plot_2, kde = True, fill = True)
            st.pyplot(fig, use_container_width = False, clear_figure = True)
        else:
            fig = plt.figure(figsize = (6, 3), dpi = 100)
            sns.countplot(data = origin, x = col_plot_2, hue = yname)
            st.pyplot(fig, use_container_width = False, clear_figure = True)