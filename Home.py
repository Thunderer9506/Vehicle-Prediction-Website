import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import io

st.set_page_config(
    page_title="Home",
    page_icon="👋",
)

st.sidebar.success("Go to Prediction Page to Predict the price of your Car")

cleaned_data = pd.read_csv('cleanedData.csv')

st.title("Vehicle Prdiction Model")

st.write("## Statistics")

st.write('### Shape of the data: ')
st.write(f'Rows : {cleaned_data.shape[0]}')
st.write(f'Cols( 17 ) : `{", ".join(list(cleaned_data.columns))}`')

buffer = io.StringIO()
cleaned_data.info(buf=buffer)
info = buffer.getvalue()

st.divider()

st.write('### Data Info')
st.text(info[37:])

st.divider()

st.write('### Data Described')
st.write(cleaned_data.describe())

st.write('## Exploratory Data Analysis')

st.write('### Price Distribution')

fig = plt.figure(figsize=(7,4))
ax = sns.histplot(cleaned_data['price'])
ax.bar_label(ax.containers[0])
ax.set_xlabel("Price ( In USD $ )",fontdict={"weight":"bold","size":14})
ax.set_ylabel("Count",fontdict={"weight":"bold","size":14})
fig.tight_layout()

st.pyplot(fig)

st.write('### Value Count of each Car Maker')

fig = plt.figure(figsize=(8,6))

ax = sns.barplot(data=cleaned_data['make'].value_counts(),errorbar=None,estimator="sum",orient="y",width=1,gap=0.2)
ax.bar_label(ax.containers[0])
ax.set_xlabel("Count",fontdict={"weight":"bold","size":14})
ax.set_ylabel("Maker's Name",fontdict={"weight":"bold","size":14})

fig.tight_layout()
st.pyplot(fig)

st.write('### Fuel Distribution')

fig = plt.figure(figsize=(8,4))
ax = sns.barplot(cleaned_data['fuel'].value_counts(),errorbar=None,orient='y')
ax.bar_label(ax.containers[0])
ax.set_xlabel("Count",fontdict={"weight":"bold","size":14})
ax.set_ylabel("Fuel Type",fontdict={"weight":"bold","size":14})
fig.tight_layout()
st.pyplot(fig)

st.write('### Body Distribution')

fig = plt.figure(figsize=(8,4))
ax = sns.barplot(cleaned_data['body'].value_counts(),errorbar=None,orient='y')
ax.bar_label(ax.containers[0],fontsize = 10)
ax.set_xlabel("Count",fontdict={"weight":"bold","size":14})
ax.set_ylabel("Body Type",fontdict={"weight":"bold","size":14})
fig.tight_layout()
st.pyplot(fig)

st.write('### Types of Drive train')

fig = plt.figure(figsize=(7,5))

ax = sns.histplot(cleaned_data['drivetrain'])
ax.bar_label(ax.containers[0])
ax.tick_params(axis='x')
ax.set_xlabel("Drive Trains",fontdict={"weight":"bold","size":14})
ax.set_ylabel("Count",fontdict={"weight":"bold","size":14})

fig.tight_layout()
st.pyplot(fig)

st.write('### Total spending by maker (sum of prices)')

make_price = cleaned_data.groupby('make', as_index=False)['price'].sum()
fig = plt.figure(figsize=(11,8))
ax = sns.scatterplot(
    data=make_price,
    x='price',
    y='make',
    size='price',
    hue='price',
    sizes=(50, 500),
    palette='viridis',
    legend=False
)

max_price = make_price['price'].max()
ticks = np.linspace(0, max_price, 6)
ax.set_xticks(ticks)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'${x/1000:.0f}k'))

ax.set_xlabel('Total Price (USD)',fontdict={"weight":"bold","size":14})
ax.set_ylabel("Maker's Name",fontdict={"weight":"bold","size":14})
fig.tight_layout()
st.pyplot(fig)

st.divider()

st.write("## Results")
st.text('1. Avg Price range of cars are $30k - $60k')
st.write('''
         2. These are the Top 5 Most selling cars
            - JEEP
            - Hyundai
            - Dodge
            - Ford
            - Ram
         ''')
st.text('3. Gasoline Cars Performed best in the market')
st.text('4. Customer preferred SUV body type cars the most')
st.text('5. All-Wheel and 4-Wheel drive cars are used more')
st.text('6. Jeep made over $9000k+ after Hyundai and Ram which made over 4000k+ each in car selling')