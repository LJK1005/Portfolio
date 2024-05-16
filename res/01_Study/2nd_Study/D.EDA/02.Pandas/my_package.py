def my_scaler(df, df2, scale = 'ss', obj = 'oh'):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
    if scale == 'ss':
        scaler = StandardScaler()
    elif scale == 'mm':
        scaler = MinMaxScaler()
    else:
        scaler = RobustScaler()

    col_n = df.select_dtypes(['int64', 'float64']).columns
    col_o = df.select_dtypes('object').columns
    col_b = df.select_dtypes('bool').columns

    df_n = df[col_n]
    df_o = df[col_o]
    df_b = df[col_b]
    
    df2_n = df2[col_n]
    df2_o = df2[col_o]
    df2_b = df2[col_b]

    try:
        df_n = pd.DataFrame(scaler.fit_transform(df_n), columns = col_n)
        df2_n = pd.DataFrame(scaler.transform(df2_n), columns = col_n)
    except:
        pass
    
    try:
        if obj == 'oh':
            df_temp = pd.concat([df_o, df2_o], axis = 0)
            df_temp = pd.get_dummies(df_temp)
            df_o = df_temp[:len(df_o)]
            df2_o = df_temp[len(df_o):]
        else:
            df_temp = pd.concat([df_o, df2_o], axis = 0)
            for i in col_o:
                le = LabelEncoder()
                df_temp[i] = le.fit_transform(df_temp[i])
            df_o = df_temp[:len(df_o)]
            df2_o = df_temp[len(df_o):]
    except:
        pass

    df = pd.concat([df_n, df_o, df_b], axis = 1)
    df2 = pd.concat([df2_n, df2_o, df2_b], axis = 1)

    return df, df2