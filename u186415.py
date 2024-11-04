import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from datetime import time, date
import time as t

# Step 4: Add a title and subheader
st.title('This is an example of title')
st.subheader("This is an example of subheader")
st.write("This is an example of a sentence")

# Step 7: Add sidebar and emoticons
st.write("This streamlit app adds *different formats* and icons like :sunglasses: and :snow_cloud:")
st.sidebar.header("The header of the sidebar")
st.sidebar.write("*Hello*")

# Step 8: Create a dataframe and plot charts
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])
st.write("This is a line chart")
st.line_chart(chart_data)
st.write("This is an area chart")
st.area_chart(chart_data)
st.write("This is a bar chart")
st.bar_chart(chart_data)

# Step 9: Matplotlib and Seaborn plots
arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)
st.write("Example 1 of a plot with Matplotlib")
st.pyplot(fig)

penguins = sns.load_dataset("penguins")
st.dataframe(penguins[["species", "flipper_length_mm"]].sample(6))
fig = plt.figure(figsize=(9, 7))
sns.histplot(data=penguins, x="flipper_length_mm", hue="species", multiple="stack")
plt.title("Hello Penguins!")
st.write("Example of a plot with Seaborn library")
st.pyplot(fig)

# Step 11: Create maps and Plotly plots
df = pd.DataFrame(np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4], columns=['lat', 'lon'])
st.write("Example of a plot with a map")
st.map(df)

x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn
