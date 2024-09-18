import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline

# Load Hugging Face text generation model (e.g., GPT-2)
text_generator = pipeline('text-generation', model='gpt2')

# Title of the app
st.title('Data File Analysis: Generate Charts and Text')

# Step 1: File upload
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Step 2: Read the file into a DataFrame
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        st.write("Data Preview:")
        st.write(df.head())

        # Step 3: User selects column to plot
        column_to_plot = st.selectbox("Select a column to visualize", df.columns)

        # Step 4: Generate and display chart using matplotlib/seaborn
        st.write(f"Visualizing data from {column_to_plot}:")
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column_to_plot], kde=True)
        st.pyplot(plt)

        # Step 5: Generate text based on data (e.g., describe the column data)
        st.write("Generate text description based on data:")

        # Generate text input prompt
        num_records = len(df)
        prompt = f"This dataset has {num_records} records. Here's some information about the column '{column_to_plot}' with a few sample values: {df[column_to_plot].dropna().sample(3).tolist()}"

        # Use Hugging Face GPT-2 or another model to generate text
        generated_text = text_generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']

        # Display generated text
        st.write(f"Generated Description: {generated_text}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
