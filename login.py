import streamlit as st
import mysql.connector as sql
import pandas as pd

usr=""
#pswd=""
signedIn=False
def app():
    global usr
    #global pswd
    global signedIn
    st.header("SignIn :lock:")
    usr = st.text_input("Username")
    #print("Here")
    pswd = st.text_input("Password", type="password")
    A, B,C=st.beta_columns([3,2,2])
    if A.button("Submit"):
        #print("Here")
        db_connection = sql.connect(host='portfoliomanagement.c5r1ohijcswm.ap-south-1.rds.amazonaws.com',
                                    database='portfolioManagement', user='admin', password='syseng1234')
        query = "SELECT * from userlogin WHERE username='" + usr + "' and password='"+pswd+"'"
        loggedIn = pd.read_sql(query, con=db_connection)
        print("Username@login: ", usr)
        print("Password@login: ", pswd)
        print(loggedIn)
        if not loggedIn.empty:
            signedIn = True
            st.write("`LoggedIn`")
            #st.success("Successfully LoggedIn")
            C.button("Go to Dashboard")
        else:
            st.error("Oops! Your Username/Password is wrong.")
            signedIn=False




