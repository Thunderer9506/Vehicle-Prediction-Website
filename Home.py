import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# -----------------------------------------
# 🎨 Streamlit Page Configuration
# -----------------------------------------
st.set_page_config(
    page_title="Vehicle Price Analysis Dashboard",
    page_icon="🚗",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.header("📍 Navigation")
    st.success("👉 Go to Prediction Page to predict your car price")
    st.markdown("---")
    st.info("This dashboard gives a detailed overview of vehicle data and trends.")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('cleanedData.csv').drop(['Unnamed: 0'],axis=1)

cleaned_data = load_data()

# -----------------------------------------
# 🏠 HEADER SECTION
# -----------------------------------------
st.title("🚘 Vehicle Price Analysis & Insights Dashboard")
st.text(" stimate your vehicle's market value instantly using key specs like make, model, and engine type, powered " \
"by a data-driven Random Forest model.")


st.markdown("---")

# -----------------------------------------
# 📊 DATA OVERVIEW
# -----------------------------------------
st.header("📋 Dataset Overview")
st.caption("Note: All of statistics and Plots are generated from Cleaned data they may vary from original dataset")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{cleaned_data.shape[0]}")
col2.metric("Columns", f"{cleaned_data.shape[1]}")
col3.metric("Unique Makers", f"{cleaned_data['make'].nunique()}")
col4.metric("Record Year", f"{cleaned_data['year'].min()}-{cleaned_data['year'].max()}")

with st.expander("🧾 View Descriptive Statistics"):
    st.dataframe(cleaned_data.describe())

with st.expander("👀 Preview Dataset and Columns"):
    st.write(f"Columns: {list(cleaned_data.columns)}")
    st.dataframe(cleaned_data.head())

st.markdown("---")

# -----------------------------------------
# 📈 EDA WITH TABS
# -----------------------------------------
st.header("📊 Exploratory Data Analysis (EDA)")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💰 Price Distribution",
    "🏭 Car Makers",
    "⛽ Fuel Type Distribution",
    "🚙 Body Type Distribution",
    "⚙️ Drive Train Distribution",
    "💵 Total Sales by Maker"
])

# --- Tab 1: Price Distribution ---
with tab1:
    st.subheader("💰 Price Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(cleaned_data['price'], kde=True, ax=ax, color='skyblue')
    ax.set_xlabel("Price (USD)", fontdict={"weight": "bold", "size": 12})
    ax.set_ylabel("Count", fontdict={"weight": "bold", "size": 12})
    st.pyplot(fig)

# --- Tab 2: Car Makers ---
with tab2:
    st.subheader("🏭 Most Common Car Makers")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(y=cleaned_data['make'].value_counts().index[:10],
                x=cleaned_data['make'].value_counts().values[:10],
                ax=ax, palette='viridis')
    ax.set_xlabel("Count", fontdict={"weight": "bold", "size": 12})
    ax.set_ylabel("Maker", fontdict={"weight": "bold", "size": 12})
    st.pyplot(fig)

# --- Tab 3: Fuel Type ---
with tab3:
    st.subheader("⛽ Fuel Type Distribution")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(y='fuel', data=cleaned_data, palette='mako', ax=ax)
    ax.set_xlabel("Count", fontdict={"weight": "bold", "size": 12})
    ax.set_ylabel("Fuel Type", fontdict={"weight": "bold", "size": 12})
    st.pyplot(fig)

# --- Tab 4: Body Type ---
with tab4:
    st.subheader("🚙 Body Type Distribution")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(y='body', data=cleaned_data, palette='Set2', ax=ax)
    ax.set_xlabel("Count", fontdict={"weight": "bold", "size": 12})
    ax.set_ylabel("Body Type", fontdict={"weight": "bold", "size": 12})
    st.pyplot(fig)

# --- Tab 5: Drive Train ---
with tab5:
    st.subheader("⚙️ Drive Train Distribution")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(x='drivetrain', data=cleaned_data, palette='coolwarm', ax=ax)
    ax.set_xlabel("Drive Train", fontdict={"weight": "bold", "size": 12})
    ax.set_ylabel("Count", fontdict={"weight": "bold", "size": 12})
    st.pyplot(fig)

# --- Tab 6: Total Sales by Maker ---
with tab6:
    st.subheader("💵 Total Sales Value by Maker")
    make_price = cleaned_data.groupby('make', as_index=False)['price'].sum()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=make_price,
        x='price', y='make', size='price', hue='price',
        sizes=(50, 500), palette='viridis', legend=False, ax=ax
    )
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'${x/1000:.0f}k'))
    ax.set_xlabel("Total Price (USD)", fontdict={"weight": "bold", "size": 12})
    ax.set_ylabel("Maker", fontdict={"weight": "bold", "size": 12})
    st.pyplot(fig)

st.markdown("---")

# -----------------------------------------
# 📈 INSIGHTS SUMMARY
# -----------------------------------------
st.header("🧠 Key Insights & Summary")

st.markdown("""
<style>
.insights {
    font-size:16px;
    line-height:1.6;
}
.insights b {
    color:#1E90FF; /* Light blue accent */
}
</style>
<div class='insights'>
<ol>
<li>Average car prices fall between <b>$30K – $60K</b>.</li>
<li>Top 5 most popular car makers: <b>Jeep, Hyundai, Dodge, Ford, Ram</b>.</li>
<li>Gasoline-powered vehicles dominate the market.</li>
<li><b>SUVs</b> are the most preferred body type.</li>
<li>All-wheel and 4-wheel drive vehicles are widely used.</li>
<li><b>Jeep</b> generated over <b>$9000K</b> in total sales, followed by <b>Hyundai</b> and <b>Ram</b> (Approx. <b>$4000K</b> each).</li>
</ol>
</div>
""", unsafe_allow_html=True)


st.markdown("---")
st.caption("© 2025 Vehicle Insights Dashboard | Created by Shaurya Srivastava 🚗💡")
