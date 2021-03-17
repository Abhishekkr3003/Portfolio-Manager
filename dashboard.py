import streamlit as st
import login
from mysql.connector import Error
import mysql.connector as msql
import pandas as pd
import globals
import random

def app():
    #st.title("Dashboard")
    st.markdown("<h2 style='text-align: center; color: black;'>Dashboard</h2>", unsafe_allow_html=True)
    imgs=["https://www.azquotes.com/vangogh-image-quotes/84/92/Quotation-Warren-Buffett-The-best-investment-you-can-make-is-an-investment-in-84-92-34.jpg", "https://www.azquotes.com/vangogh-image-quotes/87/85/Quotation-Warren-Buffett-If-you-don-t-find-a-way-to-make-money-87-85-65.jpg",
          "https://www.azquotes.com/vangogh-image-quotes/88/3/Quotation-Warren-Buffett-Look-for-3-things-in-a-person-Intelligence-Energy-Integrity-88-3-0301.jpg", "https://www.azquotes.com/vangogh-image-quotes/68/94/Quotation-Warren-Buffett-Do-not-save-what-is-left-after-spending-but-spend-68-94-79.jpg"]
    n = random.randint(0, 3)
    st.image(imgs[n])
    db_connection = msql.connect(host='portfoliomanagement.c5r1ohijcswm.ap-south-1.rds.amazonaws.com',
                                 database='portfolioManagement', user='admin', password='syseng1234')
    query8 = "SELECT budget from userlogin WHERE username='" + login.usr + "'"
    value = pd.read_sql(query8, con=db_connection)
    value = value.values.reshape(-1, ).tolist()
    value1 = float(value[0])
    st.subheader("Current Budget: Rs `" + str(value1) + "`")
    totalBuy=st.empty()
    globals.budget=value1
    a,b=st.beta_columns([5,1])
    path1 = a.number_input("Enter The Quantity", value=value1, min_value=0.0)
    st.write("Update Budget to Rs `"+str(path1)+"`")
    if b.button("Update"):
        try:
            db_connection = msql.connect(host='portfoliomanagement.c5r1ohijcswm.ap-south-1.rds.amazonaws.com',
                                         database='portfolioManagement', user='admin', password='syseng1234')
            if db_connection.is_connected():
                print("Clicked")
                cursor = db_connection.cursor()
                cursor.execute("select database();")
                record = cursor.fetchone()
                print("You're connected to database: ", record)
                sql = "UPDATE userlogin SET budget = "+str(path1)+ " WHERE username='" + login.usr + "'"
                cursor.execute(sql, ())
                print("Record Updated")
                db_connection.commit()
                st.success("Successfully Updated your Budget")
                st.button("Refresh")
                # globals.selected2 -= 1

        except Error as e:
            print(e)
            st.error("0ops! budget: Rs" + str(path1) + " cannot be allocated.")

    db_connection = msql.connect(host='portfoliomanagement.c5r1ohijcswm.ap-south-1.rds.amazonaws.com',
                                 database='portfolioManagement', user='admin', password='syseng1234')

    query5 = "SELECT Symbol,netProfit,quantity from portfolio WHERE username='" + login.usr + "'"
    snp = pd.read_sql(query5, con=db_connection)
    st.write("\n")
    st.subheader("Current Portfolio :file_folder:")
    st.table(snp)
    st.write("\n")
    query1 = "SELECT Symbol from portfolio WHERE username='" + login.usr + "'"

    symbols = pd.read_sql(query1, con=db_connection)
    # st.table(symbols)
    query2 = "SELECT netProfit from portfolio WHERE username='" + login.usr + "'"
    sum_ = pd.read_sql(query2, con=db_connection)
    sum2 = sum_.values.reshape(-1, ).tolist()
    sum = 0
    print(sum2)
    for i in range(0, len(sum2)):
        sum = sum + (sum2[i])
    st.subheader("Your net profit is: Rs `"+str(float(sum))+"`")
    query3 = "SELECT totalCost from portfolio WHERE username='" + login.usr + "'"
    cost_ = pd.read_sql(query3, con=db_connection)
    
    cost = cost_.values.reshape(-1, ).tolist()
    ser = 0
    print(cost)
    for i in range(0, len(cost)):
        ser = ser + (cost[i])
    globals.ser = ser
    totalBuy.subheader("Current Spending: Rs `" + str(float(ser)) + "`")
    argument = int(ser / 100000)
    l,m,r=st.beta_columns([1,10,1])
    if (argument == 0):
        if (sum > 2500):
            st.markdown("You are doing GREAT! An average trader makes Rs `2500`")
            m.markdown("![Alt Text](https://media.giphy.com/media/f7GQKWSKo5ekWPUNnC/giphy.gif)")

        else:
            st.markdown("You are making below than an average trader. An average trader makes Rs `2500`")
            m.markdown("![Alt Text](https://media.giphy.com/media/3oKHW5B0wqqvBcLkdy/giphy.gif)")

    elif (argument == 1):
        if (sum > 4000):
            st.markdown("You are doing GREAT! An average trader makes Rs `4000`")
            m.markdown("![Alt Text](https://media.giphy.com/media/f7GQKWSKo5ekWPUNnC/giphy.gif)")
        else:
            st.markdown("You are making below than an average trader.An average trader makes Rs `4000`")
            m.markdown("![Alt Text](https://media.giphy.com/media/3oKHW5B0wqqvBcLkdy/giphy.gif)")

    elif (argument == 2):
        if (sum > 7000):
            st.markdown("You are doing GREAT! An average trader makes Rs `7000`")
            m.markdown("![Alt Text](https://media.giphy.com/media/f7GQKWSKo5ekWPUNnC/giphy.gif)")
        else:
            st.markdown("You are making below than an average trader.An average trader makes Rs `7000`")
            m.markdown("![Alt Text](https://media.giphy.com/media/3oKHW5B0wqqvBcLkdy/giphy.gif)")

    else:
        if (sum > 10000):
            st.markdown("You are doing GREAT! An average trader makes Rs `10000`")
            m.markdown("![Alt Text](https://media.giphy.com/media/f7GQKWSKo5ekWPUNnC/giphy.gif)")

        else:
            st.markdown("You are making below than an average trader. An average trader makes Rs `10000`")
            m.markdown("![Alt Text](https://media.giphy.com/media/3oKHW5B0wqqvBcLkdy/giphy.gif)")

    st.write("\n")
    st.subheader("Delete From Portfolio")
    symbolSelected = st.selectbox("In your Portfolio", symbols.values.reshape(-1, ).tolist())
    if st.button("Delete"):
        path = symbolSelected
        cursor = db_connection.cursor()
        cursor.execute("select database();")
        record = cursor.fetchone()
        print("You're connected to database: ", record)
        sql = "DELETE from portfolio WHERE Symbol='" + path + "' and username='" + login.usr + "'"
        cursor.execute(sql)
        print("Record Deleted")
        # st.balloons()
        db_connection.commit()
        st.success(path + " Is Successfully Deleted From Your Portfolio")
        # st.dataframe(snp.style.highlight_min(axis=0))
