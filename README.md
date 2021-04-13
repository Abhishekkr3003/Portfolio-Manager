# Portfolio Management System

## Introduction
This system aims to predict stocks price value using machine learning. This project is to solve the economic dilemma created by individuals who want to invest in the market.

## Functional and Non-Functional Requirements
* __Funtional Requrements__: The system shall be able to generate an approximate share price. The system shall show the current status of the stock market by providing charts and graphs.
* __Non-Functional Requirements__: The system shall collect accurate data from the source website consistently. The system provides a simplistic user interface that guides and assists a new user in understanding the stock market with ease.

## Components And Their Relationship
The system has four components, namely, 
* User			
* Portfolio		
* Company
* Company's date wise data

Components and their relationship diagram:
![Alt Text](https://lh4.googleusercontent.com/jZWPDKQ4r1wkZeQuUbJn54RRjTk8WnnWwIq5vWhHf6sSgBQj4Uq01T524QyJ7fYiufXa0I8eYVJjm4wWpcD2Lb5wpIljsqCcF0zVqSEQ=s1600)


## Tools and Technology
The project is build using Python with MySQL as a database. For predictions, we will use the Keras  library, LSTM RNN and data analysis will be done through NumPy, Pandas and Matplotlib. The frontend part is implemented through Streamlit.


 
## Implementation
The system consists of databases tables of the four components mentioned above. All the data will be stored and managed with Arrays and Matrices. The user will log in and allot a certain amount to his wallet. He can then select the sector or specific company for which he wants predictions. He can add as many as companies he wants to his portfolio. He can also manually edit the portfolio and monitor other stocks with the help of charts and graphs. The user can also view various companies' information and average returns on the portfolio, which will be implemented using web scraping and python libraries.

Many variables could affect the share market, directly or indirectly. For this chaotic system, the neural network approach is suitable because we need to show the correct output for the neural network's given inputs. With a sufficient amount of training, the network will mimic the function. Another advantage of the neural network is that it will learn to ignore inputs that do not contribute to the output. There is a training phase in our proposed system where some parameters named weights are found from this section, and Backpropagation Algorithm through an LSTM RNN is used for this training phase. 
