import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm

#import streamlit as st
import plotly.express as px
from datetime import datetime
#from streamlit_extras.metric_cards import style_metric_cards
#from streamlit_extras.grid import grid

source2_df = pd.DataFrame(columns=['Name', 'Age'])
source2_df.loc[len(source2_df)] = [1, 2]
source2_df.loc[len(source2_df)] = [3, 4]
source2_df.iat[-1, 1] = 5

print(source2_df)
                     
