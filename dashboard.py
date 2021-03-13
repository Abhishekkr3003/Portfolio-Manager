import streamlit as st
import login

def app():
    st.header("Dashboard")
    if st.button("Press me"):
        print("Username: ", login.usr)