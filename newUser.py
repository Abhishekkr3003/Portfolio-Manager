import streamlit as st
import mysql.connector as msql
from mysql.connector import Error
import login

signedIn=False
def app():
    st.header("SignUp :moneybag:")
    name=st.text_input("Name")
    email=st.text_input("E-Mail")
    username=st.text_input("Username")
    password=st.text_input("Password", type="password")
    A,C,B=st.beta_columns(3)
    if A.button("Submit"):
        try:
            conn = msql.connect(host='portfoliomanagement.c5r1ohijcswm.ap-south-1.rds.amazonaws.com',
                                database='portfolioManagement', user='admin', password='syseng1234')
            if conn.is_connected():
                cursor = conn.cursor()
                cursor.execute("select database();")
                record = cursor.fetchone()
                print("You're connected to database: ", record)
                sql = "INSERT INTO userlogin VALUES (%s,%s,%s,%s)"
                cursor.execute(sql, (name, email, username, password))
                print("Record inserted")
                conn.commit()
                login.usr=username;
                login.signedIn = True
                st.success("Successfully SignedUp")
                B.button("Go to Dashboard")
        except Error as e:
            print("Error while connecting to MySQL", e)