import streamlit as st
from snowflake.snowpark.context import get_active_session
import pandas as pd
import matplotlib.pyplot as plt
from snowflake.cortex import complete

# Establish connection using the name defined in your secrets.toml
conn = st.connection("snowflake")

# Simplified query without extra arguments that might trigger TypeErrors
df = conn.query("SELECT * FROM REVIEWS_C_SENTIMIENTO")

# To verify it worked
if df is not None:
    st.dataframe(df.head())

# App title and sidebar filters
st.title("Product Intelligence Dashboard")
products = df['PRODUCT'].unique()
selected_products = st.multiselect("Select Products:", options=products, default=products)
filtered_df = df[df['PRODUCT'].isin(selected_products)]

# Data preview
st.subheader("Data Preview")
st.dataframe(filtered_df.head()) # una muestra del dataframe que esta filtrado

# Visualization: Average Sentiment by Region
st.subheader("Average Sentiment by Region")
region_sentiment = filtered_df.groupby("REGION")['SENTIMENT_SCORE'].mean().sort_values()
fig, ax = plt.subplots()
region_sentiment.plot(kind="barh", ax=ax, title="Average Sentiment by Region")
ax.set_xlabel("Sentiment Score")
st.pyplot(fig)

# Highlight Delivery Issues
st.subheader("Delivery Issues by Region and Product")
grouped_issues = filtered_df.groupby(['REGION', 'PRODUCT'])[['STATUS', 'SENTIMENT_SCORE']].first().reset_index()
st.dataframe(grouped_issues) 

# Chatbot assistant
st.subheader("Ask Questions About Your Data")
user_question = st.text_input("Enter your question here:")
df_string = df.to_string(index=False) # convierte toda la tabla en una cadena de texto (mini RAG?)
if user_question: 
    response = complete(model="mixtral-8x7b", prompt=f"Answer this question using the dataset: {user_question} <context>{df_string}</context>", session=session)
    st.write(response) # acá esta utilizando unos de los modelos dentro de cortex.
    # en tal caso se descuenta de los creditos de snowflake. 
