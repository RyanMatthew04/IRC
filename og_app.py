import streamlit as st
import pandas as pd
from fuzzywuzzy import process, fuzz

# Streamlit App Layout
st.title("Testing Name Match Logic")

# File Upload: Upload dataset with "name" column
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv", "xlsx"])

if uploaded_file:
    # Read the dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Ensure the dataset has a 'name' column
    if 'name' not in df.columns:
        st.error("Dataset must contain a 'name' column.")
    else:
        # Display the dataset
        st.write("Dataset Preview:")
        st.write(df.head())

        # Text Input for company name to match
        new_name = st.text_input("Enter the company name to match:")

        if new_name:
            # Get list of company names
            master_names = df['name'].tolist()

            # Perform fuzzy matching
            top_matches = process.extract(new_name, master_names, scorer=fuzz.partial_ratio, limit=10)

            # Display the results
            st.write("Top 10 matches:")
            for match, score in top_matches:
                st.write(f"{match}")

