import streamlit as st
import login
import sectorWise
import newUser
import dashboard
import logout
import companyWise
from multiapp import MultiApp

def print_hi(name):
    print(f'Hi, {name}')

if __name__ == '__main__':
    st.markdown("<h1 style='text-align: center; color: black;'>Portfolio Manager</h1>", unsafe_allow_html=True)
    app = MultiApp()
    print(app)
    if not login.signedIn:
        app.add_app("SignIn", login.app)
        app.add_app("SignUp", newUser.app)
    else:
        app.add_app("Dashboard", dashboard.app)
        app.add_app("Sector Wise Predictoin",sectorWise.app)
        app.add_app("Company Wise Prediction", companyWise.app)
        app.add_app("Logout", logout.app)

    app.run()

