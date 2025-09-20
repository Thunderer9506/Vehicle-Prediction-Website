import streamlit as st
import pandas as pd
import io

cleaned_data = pd.read_csv('cleaned_dataset.csv')

st.title("Vehicle Prdiction Model")

st.write("## Analytics")

st.write('### Shape of the data: ')
st.write('Rows : 894')
st.write(f'Cols( 17 ) : `{", ".join(['name','description','make','model','year','price','engine','cylinders','fuel','mileage','transmission','trim','body','doors','exterior_color','interior_color','drivetrain'])}`')

buffer = io.StringIO()
cleaned_data.info(buf=buffer)
info = buffer.getvalue()

st.divider()

st.write('### Data Info')
st.text(info[37:])

st.divider()

st.write('### Data Described')
st.write(cleaned_data.describe())
