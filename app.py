import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm

import streamlit as st
import plotly.express as px
from datetime import datetime
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.grid import grid
import altair as alt

def build_sidebar():
    st.image("images/ProfilePic.png")
    ticker_list = pd.read_csv("list.csv", index_col=0)
    ticker = st.selectbox(label="Select cryptocurrency", options=ticker_list, placeholder='Tickers')

    start_date = st.date_input("From", format="DD/MM/YYYY", value=datetime(2025,1,2))
    end_date = st.date_input("Until", format="DD/MM/YYYY", value="today")

    if ticker:
        prices = yf.download(ticker, start=start_date, end=end_date)['Close']

                    
        prices.columns = prices.columns

        return ticker, prices
    return None, None


def build_main(ticker, prices):
    # Inputting parameters
    r0 = int(len(prices)*0.1)
    # Lags for the Augmented Dickey-Fuller Test
    adf_lags = 3
    # Crital value of the right-tailed ADF-Test from paper (%95)
    crit = 1.49 #Finding using bootstraping in article
    # Transforming Data - Make algorithm more robust log prices used
    log_prices = np.array(np.log(prices))
    delta_log_prices = log_prices[1:] - log_prices[:-1]
    n = len(delta_log_prices)
    BSADF = np.array([])
    Bubbles_df = pd.DataFrame(columns=['Start', 'End', 'Count'])
    
    
    #Calculating ADF stats
    for r2 in range(r0,n):
        ADFS = np.array([])
        for r1 in range(0, r2-r0+1):
            # Ensure X_0 is 1-dimensional
            X_0 = log_prices[r1:r2+1].flatten()
            X = pd.DataFrame()
            X[0] = X_0
            for j in range(1, adf_lags+1):
                # Ensure delta_log_prices is 1-dimensional before slicing for appending
                X[j] = np.append(np.zeros(j), delta_log_prices[r1:r2+1-j].flatten())
            # X = np.array(X) # No longer need to convert back to numpy here
            y = delta_log_prices[r1:r2+1].flatten() # Ensure y is 1-dimensional
            # If X is a pandas DataFrame, sm.add_constant will work directly
            reg = sm.OLS(y, sm.add_constant(X))
            res = reg.fit()
            ADFS = np.append(ADFS, res.params[1]/res.bse[1])
        BSADF = np.append(BSADF, max(ADFS))
        if len(BSADF)>1 and BSADF[-1] > crit and BSADF[-2] <= crit:
            Bubbles_df.loc[len(Bubbles_df)] = [prices.index[r2], prices.index[r2], len(Bubbles_df)]
        if BSADF[-1] > crit:
            Bubbles_df.iat[-1, 1] = prices.index[r2]

    data = {"Date" : prices.index[r0+1:],
            "BSADF" : BSADF,
            "Critical Value" :np.ones(len(BSADF))*crit,
            "Price" : prices[r0+1:].values.flatten() }
    graph2 = pd.DataFrame(data)

    mygrid = grid(2 ,2 ,2 ,2 ,2 , 2, vertical_align="top")
    
    c = mygrid.container(border=True)
    c.subheader(ticker, divider="red")

    colA, colB = c.columns(2)

    if BSADF[-1] > crit: BubNow = 'Yes'
    else: BubNow = "No"
    colA.metric(label="BSADF", value=f"{BSADF[-1]}")
    colB.metric(label="Bubble?", value=BubNow)
    style_metric_cards(background_color='rgba(255,255,255,0)')

    line_chart = alt.Chart(graph2).mark_line().encode(
    alt.Y('Price:Q').scale(zero=False),
    x='Date:T',
    ) #.properties(height=500)

    rect_chart = alt.Chart(Bubbles_df).mark_rect().encode(
    x="Start:T",
    x2="End:T",
    color=alt.Color("Count:N").title("Event").scale(scheme="lightgreyred")
    ) #.properties(height=500)

    col1, col2 = st.columns(2, gap='large')
    with col1:
        st.subheader("Prices")
       # st.line_chart(prices[r0+1:], height=600)
        st.altair_chart(rect_chart + line_chart)

    with col2:
        st.subheader("BSADF")
        st.line_chart(graph2, x='Date', y=["BSADF", "Critical Value"])
#plt.rc('xtick',labelsize = 8)
#plt.plot(prices.index[r0+1:],BSADF)
#plt.plot(prices.index[r0+1:],np.ones(len(BSADF))*crit)

#plt.show()

st.set_page_config(layout="wide")

with st.sidebar:
    ticker, prices = build_sidebar()

st.title('Crypto Dashboard')
if ticker:
    build_main(ticker, prices)
