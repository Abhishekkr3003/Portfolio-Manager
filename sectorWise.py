from datetime import date

import streamlit as st
import login
import pandas as pd
import mysql.connector as msql
import math
import main
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from keras.models import load_model
from mysql.connector import Error
import globals
import pathlib


@st.cache(allow_output_mutation=True)
def PredictorModel(sector_):
    sector=sector_
    db_connection = msql.connect(host='portfoliomanagement.c5r1ohijcswm.ap-south-1.rds.amazonaws.com',
                                database='portfolioManagement', user='admin', password='syseng1234')
    query = "SELECT Symbol from companies WHERE Sector='" + sector + "'"
    sectorWiseCompanies = pd.read_sql(query, con=db_connection)

    for ind in sectorWiseCompanies.index:
        db_connection = msql.connect(host='portfoliomanagement.c5r1ohijcswm.ap-south-1.rds.amazonaws.com',
                                    database='portfolioManagement', user='admin', password='syseng1234')
        query = "SELECT * from companyDateWise WHERE Symbol='" + sectorWiseCompanies['Symbol'][ind] + "'"
        eachCompany = pd.read_sql(query, con=db_connection)
        data = eachCompany.filter(['Close'])
        # Converting the dataframe to a numpy array
        dataset = data.values
        # Get /Compute the number of rows to train the model on
        training_data_len = math.ceil(len(dataset) * .75)

        # Scale the all of the data to be values between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        file = pathlib.Path('models/' + sectorWiseCompanies['Symbol'][ind] + '.model/saved_model.pb')

        if not file.exists ():
            print('models/'+sectorWiseCompanies['Symbol'][ind] + '.model'+ "NOT FOUND**************************")
        else:
            print('models/' + sectorWiseCompanies['Symbol'][ind] + '.model' + "FOUND**************************")
        if not file.exists ():
            # Create the scaled training data set
            train_data = scaled_data[0:training_data_len, :]
            x_train = []
            y_train = []
            for i in range(30, len(train_data)):
                x_train.append(train_data[i - 30:i, 0])
                y_train.append(train_data[i, 0])

            #print("x-train: ", x_train)
            #print("y-train: ", y_train)
            # Convert x_train and y_train to numpy arrays
            x_train, y_train = np.array(x_train), np.array(y_train)

            # Reshape the data into the shape accepted by the LSTM
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            # Build the LSTM network model
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dense(units=25))
            model.add(Dense(units=1))

            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train the model
            model.fit(x_train, y_train, batch_size=50, epochs=10)
            model.save('models/'+sectorWiseCompanies['Symbol'][ind] + '.model')
    df2 = pd.DataFrame(columns=['Symbol'])
    selected_columns = sectorWiseCompanies["Symbol"]
    sectorWiseCompanies.columns = df2.columns.tolist()
    df2 = df2.append(sectorWiseCompanies)

    ## OR

    # df2 = pd.concat([sectorWiseCompanies, df2])

    sLength = len(df2['Symbol'])
    df2['Predictions'] = pd.Series(np.random.randn(sLength), index=df2.index)

    for ind in sectorWiseCompanies.index:
        db_connection = msql.connect(host='portfoliomanagement.c5r1ohijcswm.ap-south-1.rds.amazonaws.com',
                                    database='portfolioManagement', user='admin', password='syseng1234')
        query = "SELECT * from companyDateWise WHERE Symbol='" + sectorWiseCompanies['Symbol'][ind] + "'"
        eachCompany = pd.read_sql(query, con=db_connection)
        new_df = eachCompany.filter(['Close'])
        loaded_model = load_model('models/'+sectorWiseCompanies['Symbol'][ind] + '.model')
        last_30_days = new_df[-30:].values
        # Scale the data to be values between 0 and 1
        last_30_days_scaled = scaler.transform(last_30_days)
        # Create an empty list
        X_test = []
        # Append teh past 1 days
        X_test.append(last_30_days_scaled)
        # Convert the X_test data set to a numpy array
        X_test = np.array(X_test)
        # Reshape the data
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        # Get the predicted scaled price
        pred_price = loaded_model.predict(X_test)
        # undo the scaling
        pred_price = scaler.inverse_transform(pred_price)
        df2['Predictions'][ind] = pred_price
    #print(df2)
    #df2.set_index('Symbol', inplace=True)
    return df2


def app():
    if globals.selected==-1:
        st.header("Sector Wise Prediction")
        db_connection = msql.connect(host='portfoliomanagement.c5r1ohijcswm.ap-south-1.rds.amazonaws.com',
                                    database='portfolioManagement', user='admin', password='syseng1234')
        query = "select distinct sector from companies"
        result = pd.read_sql(query, con=db_connection)
        #print(result)
        global sector
        sector=st.selectbox("Select the Sector", result.stack().tolist())
        globals.sector=sector
        #st.write(globals.selected)
        if st.button("Predict Tommorow's Price of "+sector):
            dataf=PredictorModel(sector)
            st.table(dataf)
            globals.df2=dataf
            if 'Symbol' in globals.df2.columns:
                globals.df2.set_index('Symbol', inplace=True)
            globals.selected=0
            st.button("Next")


    else:
        #st.write(globals.df2)
        #A,B=st.beta_columns(2)
        db_connection = msql.connect(host='portfoliomanagement.c5r1ohijcswm.ap-south-1.rds.amazonaws.com',
                                    database='portfolioManagement', user='admin', password='syseng1234')
        query = "select distinct Symbol from companies where Sector='" + globals.sector + "'"
        result = pd.read_sql(query, con=db_connection)
        st.subheader("Insert Into Portfolio :shopping_bags:")
        symbol_ = st.selectbox("Select the "+sector+"'s Company", result.stack().tolist())
        st.write("Do you want to insert `"+symbol_+"` in your Portfolio")
        if st.button("Insert"):
            st.markdown(":robot_face: We are adding " + symbol_ + " to your portfolio :robot_face:")
            try:
                db_connection = msql.connect(host='portfoliomanagement.c5r1ohijcswm.ap-south-1.rds.amazonaws.com',
                                             database='portfolioManagement', user='admin', password='syseng1234')
                if db_connection.is_connected():
                    print("Clicked")
                    cursor = db_connection.cursor()
                    cursor.execute("select database();")
                    record = cursor.fetchone()
                    print("You're connected to database: ", record)
                    sql = "INSERT INTO portfolio VALUES (%s,%s,%s,%s)"
                    print(login.usr, symbol_, globals.df2._get_value(symbol_, 'Predictions') , date.today())
                    cursor.execute(sql, (login.usr, symbol_, str(globals.df2._get_value(symbol_, 'Predictions')) , date.today()))
                    print("Record inserted")
                    #st.balloons()
                    db_connection.commit()
                    st.success(symbol_ + " successfully added to your portfolio")

            except Error as e:
                print(e)
                st.error("0ops! "+symbol_+" Already in your database")
        A,B,C =st.beta_columns(3)
        if A.button("See All Sectors"):
            globals.selected -= 1
            B.markdown(":arrow_right: :arrow_right: :arrow_right: :arrow_right: :arrow_right: :arrow_right:")
            C.button("Press Here")