from datetime import date
import streamlit as st
import dashboard
import login
import pandas as pd
import mysql.connector as msql
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from keras.models import load_model
import globals
import pathlib
from mysql.connector import Error
import requests
import pandas as pd
from yahoo_fin import stock_info as si
from pandas_datareader import DataReader
import numpy as np

#@st.cache
def PredictorModel(symbol_):
    symbol =symbol_
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
        print('models/' + symbol + '.model' + "NOT FOUND**************************")
    else:
        print('models/' + symbol + "FOUND**************************")
    # Create the scaled training data set
    if not file.exists():
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
    loaded_model2 = load_model('models/' + symbol + '.model')
    test_data = scaled_data[training_data_len - 30:, :]

    # Create the x_test and y_test data sets
    x_test = []
    y_test = dataset[training_data_len:,
             :]  # Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
    for i in range(30, len(test_data)):
        x_test.append(test_data[i - 30:i, 0])

    # Convert x_test to a numpy array
    x_test = np.array(x_test)

    # Reshape the data into the shape accepted by the LSTM
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # #Getting the models predicted price values
    predictions = loaded_model2.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    # Visualize the data
    plt.figure(figsize=(16, 8))
    plt.title(symbol_)
    plt.xlabel('Days', fontsize=18)
    plt.ylabel('Close Price', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Past Data', 'Actual', 'Predictions'], loc='lower right')
    st.write("Tomorrows predicted price for " + symbol_ + " is ", predRes)
    st.write("Past 1 year trends and Accurancy:")
    st.pyplot()
    return predRes

def app():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    if globals.selected2==-1:
        st.header("Company Wise Prediction")
        db_connection = msql.connect(host='portfoliomanagement.c5r1ohijcswm.ap-south-1.rds.amazonaws.com',
                                    database='portfolioManagement', user='admin', password='syseng1234')
        query = "select distinct Symbol from companies"
        result = pd.read_sql(query, con=db_connection)
        globals.symbol_=st.selectbox("Select the Company", result.stack().tolist())
        print("OutsideBlock")
        if st.button("Predict For "+globals.symbol_):
            globals.predRes = PredictorModel(globals.symbol_)
            #st.write("Tomorrows predicted price for " + globals.symbol_ + " is ", globals.predRes)
            globals.selected2 = 0
            st.button("Next")
    else:

        db_connection = msql.connect(host='portfoliomanagement.c5r1ohijcswm.ap-south-1.rds.amazonaws.com',
                                     database='portfolioManagement', user='admin', password='syseng1234')
        st.subheader("Insert Into Portfolio :shopping_bags:")
        symbol_=globals.symbol_
        quantity = st.number_input("Enter The Quantity", value=1, min_value=1)
        query = "select Close from companyDateWise where Symbol='" + symbol_ + "' and Date=(SELECT MAX(Date) FROM companyDateWise WHERE Symbol='" + symbol_ + "')"
        resdf = pd.read_sql(query, con=db_connection)
        lastClose = resdf.at[0, 'Close']
        netPofit = (globals.predRes - lastClose) * quantity
        totalCost = lastClose * quantity
        pred = round(globals.predRes,2)
        st.write("Do you want to add `" + str(quantity) + "` of  `" + symbol_ + "` at Current Price `Rs " + str(
            lastClose) + "` and Predicted Price `Rs " + str(pred) + "` in your Portfolio")
        if st.button("Insert"):
            print(totalCost)
            print(globals.ser)
            print(globals.budget)
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
                        print(login.usr, symbol_, globals.predRes, date.today(), quantity,
                              netPofit, totalCost)
                        cursor.execute(sql, (
                        login.usr, symbol_, str(globals.predRes), date.today(),
                        str(quantity), str(netPofit), str(totalCost)))
                        print("Record inserted")
                        # st.balloons()
                        db_connection.commit()
                        st.success(symbol_ + " successfully added to your portfolio")
                        #globals.selected2 -= 1

                except Error as e:
                    print(e)
                    st.error("0ops! " + symbol_ + " is already in your Portfolio")
                    #globals.selected2 -= 1
        A, C = st.beta_columns(2)
        if A.button("Back"):
            globals.selected2 -= 1
            C.button("Refresh")

        st.subheader("More Details of `"+globals.symbol_+"` :information_source:")
        query = "select * from companies where Symbol='" + globals.symbol_ + "'"
        resdf = pd.read_sql(query, con=db_connection)
        st.write("Company's Name: `"+str(resdf.iloc[0]['Company_Name'])+"`")
        st.write("Sector: `" + str(resdf.iloc[0]['Sector']) + "`")
        st.write("Series: `" + str(resdf.iloc[0]['Series']) + "`")
        st.write("ISIN Code: `" + str(resdf.iloc[0]['ISIN_Code']) + "`")
        details = si.get_stats(globals.symbol_ + ".NS")
        details_new = details.rename(columns={'Attribute': 'Info'})
        details_new.dropna(inplace=True)
        st.table(details_new)
