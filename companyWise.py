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
#print(find_files("smpl.htm","D:"))

@st.cache
def PredictorModel(symbol_):

    from datetime import date

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
        model.fit(x_train, y_train, batch_size=50, epochs=10)
        model.save('models/'+symbol + '.model')


    df2 = pd.DataFrame(columns=['Symbol'])


    sLength = len(df2['Symbol'])
    df2['Predictions'] = pd.Series(np.random.randn(sLength), index=df2.index)

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
    return predRes

def app():
    st.header("Company Wise Prediction")
    db_connection = msql.connect(host='portfoliomanagement.c5r1ohijcswm.ap-south-1.rds.amazonaws.com',
                                database='portfolioManagement', user='admin', password='syseng1234')
    query = "select distinct Symbol from companies"
    result = pd.read_sql(query, con=db_connection)
    globals.symbol_=st.selectbox("Select the Sector", result.stack().tolist())
    print("OutsideBlock")
    if st.button("Predict And Add "+globals.symbol_+" To My Portfolio"):
        print("insideBlock")
        globals.predRes=PredictorModel(globals.symbol_)
        st.write("Tomorrows predicted price for " + globals.symbol_ + " is ", globals.predRes)
        st.markdown(":robot_face: We are adding "+globals.symbol_+" to your portfolio :robot_face:")
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
                cursor.execute(sql, (login.usr, globals.symbol_, globals.predRes, date.today()))
                print("Record inserted")
                st.balloons()
                db_connection.commit()
                st.success(globals.symbol_ + " successfully added to your portfolio")
        except Error as e:
            st.error("0ops! "+globals.symbol_+" Already in your database")
    if st.button("Tommorow's Expected Price Of "+globals.symbol_):
        globals.predRes = PredictorModel(globals.symbol_)
        st.write("Tomorrows predicted price for " + globals.symbol_ + " is ", globals.predRes)
    st.button("Reset")
    print("After block")