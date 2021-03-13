import streamlit as st
import login

def app():
    A,B = st.beta_columns(2)
    logout=A.text_input("To logout, type `log me out`")
    if logout=="log me out":
        login.signedIn=False
        login.usr=""
        st.success("Successfully Logged Out")
        B.button("SignIn Back :)")