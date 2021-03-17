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
from yahoo_fin import stock_info as si


def PredictorModel(sector_):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    cols = ['Symbol', 'Predictions']
    df2 = pd.DataFrame(columns=cols)
    sector = sector_
    db_connection = msql.connect(host='portfoliomanagement.c5r1ohijcswm.ap-south-1.rds.amazonaws.com',
                                database='portfolioManagement', user='admin', password='syseng1234')
    query = "SELECT Symbol from companies WHERE Sector='" + sector + "'"
    symbol_ = pd.read_sql(query, con=db_connection)
    ################################################
    # Create the title 'Portfolio Adj Close Price History
    title = 'Portfolio Adj. Close Price History    '  # Get the stocks
    # my_stocks = df#Create and plot the graph
    plt.figure(figsize=(
    12.2, 4.5))  # width = 12.2in, height = 4.5# Loop through each stock and plot the Adj Close for each day
    ############################################
    symbols = symbol_['Symbol'].tolist()
    print(symbols)
    for symbol in symbols:
        print(symbol)
        db_connection = msql.connect(host='portfoliomanagement.c5r1ohijcswm.ap-south-1.rds.amazonaws.com',
                                    database='portfolioManagement', user='admin', password='syseng1234')
        query = "SELECT * from companyDateWise WHERE Symbol='" + symbol + "'"
        eachCompany = pd.read_sql(query, con=db_connection)
        data = eachCompany.filter(['Close'])
        # Converting the dataframe to a numpy array
        dataset = data.values
        # Get /Compute the number of rows to train the model on
        training_data_len = math.ceil(len(dataset) * .75)

        # Scale the all of the data to be values between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        file = pathlib.Path('models/' + symbol + '.model/saved_model.pb')
        if not file.exists():
            print('models/' + symbol + '.model' + "NOT FOUND")
        else:
            print('models/' + symbol + "FOUND")
        # Create the scaled training data set
        if not file.exists():
        # Create the scaled training data set
            train_data = scaled_data[0:training_data_len, :]
            x_train = []
            y_train = []
            for i in range(30, len(train_data)):
                x_train.append(train_data[i - 30:i, 0])
                y_train.append(train_data[i, 0])

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
            model.fit(x_train, y_train, batch_size=50, epochs=100)
            model.save('models/'+symbol + '.model')

        db_connection = msql.connect(host='portfoliomanagement.c5r1ohijcswm.ap-south-1.rds.amazonaws.com',
                                    database='portfolioManagement', user='admin', password='syseng1234')
        query = "SELECT * from companyDateWise WHERE Symbol='" + symbol + "'"
        eachCompany = pd.read_sql(query, con=db_connection)
        new_df = eachCompany.filter(['Close'])
        loaded_model = load_model('models/'+symbol + '.model')
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
        predRes = pred_price.item(0)
        df2 = df2.append({'Symbol': symbol, 'Predictions': predRes}, ignore_index=True, )
        print("Tomorrows predicted price for " + symbol + " is ", predRes)
        plt.plot(data, label=data)
    print(df2)
    col_one_list = df2['Symbol'].tolist()
    print(col_one_list)
    plt.xlabel('Day', fontsize=18)
    plt.ylabel('Adj. Price INR (Rs.)', fontsize=18)
    plt.legend(col_one_list, bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.show()
    st.table(df2)
    st.write("Past 1 year trends:")
    st.pyplot()

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
        quantity = st.number_input("Enter The Quantity", value=1, min_value=1)

        query = "select Close from companyDateWise where Symbol='"+symbol_+"' and Date=(SELECT MAX(Date) FROM companyDateWise WHERE Symbol='"+symbol_+"')"
        resdf = pd.read_sql(query, con=db_connection)
        lastClose=resdf.at[0,'Close']
        netPofit=(globals.df2._get_value(symbol_, 'Predictions')-lastClose)*quantity
        totalCost=lastClose*quantity
        pred=round(globals.df2._get_value(symbol_, 'Predictions'))
        st.write("Do you want to add `" + str(quantity) + "` of  `" + symbol_ + "` at Current Price `Rs "+str(lastClose)+"` and Predicted Price `Rs "+str(float(pred))+"` in your Portfolio")
        if st.button("Insert"):
            if totalCost + globals.ser > globals.budget:
                st.error("Oops! You Are Exceeding The Budget")
                st.write("`Quick Help:` You can increase the budget from dashboard.")
            else:
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
                        sql = "INSERT INTO portfolio VALUES (%s,%s,%s,%s,%s,%s,%s)"
                        print(login.usr, symbol_, globals.df2._get_value(symbol_, 'Predictions') , date.today(), quantity, netPofit, totalCost)
                        cursor.execute(sql, (login.usr, symbol_, str(globals.df2._get_value(symbol_, 'Predictions')) , date.today(), str(quantity), str(netPofit), str(totalCost)))
                        print("Record inserted")
                        #st.balloons()
                        db_connection.commit()
                        st.success(symbol_ + " successfully added to your portfolio")

                except Error as e:
                    print(e)
                    st.error("0ops! "+symbol_+" Is Already In Your Portfolio")

        A,C =st.beta_columns(2)
        if A.button("Back"):
            globals.selected -= 1
            C.button("Refresh")
        st.subheader("More Details of `" + symbol_ + "` :information_source:")
        query = "select * from companies where Symbol='" + symbol_ + "'"
        resdf = pd.read_sql(query, con=db_connection)
        st.write("Company's Name: `" + str(resdf.iloc[0]['Company_Name']) + "`")
        st.write("Sector: `" + str(resdf.iloc[0]['Sector']) + "`")
        st.write("Series: `" + str(resdf.iloc[0]['Series']) + "`")
        st.write("ISIN Code: `" + str(resdf.iloc[0]['ISIN_Code']) + "`")
        details = si.get_stats(symbol_ + ".NS")
        details_new = details.rename(columns={'Attribute': 'Info'})
        details_new.dropna(inplace=True)
        st.table(details_new)

