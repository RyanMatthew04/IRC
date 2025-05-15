import streamlit as st
import pandas as pd
from fuzzywuzzy import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Testing Name Match logic")

master_file = st.file_uploader("Upload Master Dataset", type=["csv", "xlsx"], key="master")
test_file = st.file_uploader("Upload Test Dataset", type=["csv", "xlsx"], key="test")

if master_file and test_file:
    def load_file(file):
        return pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

    master_df = load_file(master_file)
    test_df = load_file(test_file)

    if 'Master_Code' not in master_df.columns or 'Master_Name' not in master_df.columns:
        st.error("Master dataset must contain 'Master_Code' and 'Master_Name' columns.")
    elif 'Buyer_Name' not in test_df.columns:
        st.error("Test dataset must contain 'Buyer_Name' column.")
    else:
        st.success("Both files loaded successfully.")

        master_df['Master_Name_Lower'] = master_df['Master_Name'].str.lower().str.strip()
        test_df['Buyer_Name_Lower'] = test_df['Buyer_Name'].str.lower().str.strip()

        all_names = master_df['Master_Name_Lower'].tolist()
        vectorizer = TfidfVectorizer().fit(all_names + test_df['Buyer_Name_Lower'].tolist())
        tfidf_master = vectorizer.transform(all_names)

        buyer_codes = []
        updated_buyer_names = []

        st.subheader("Manually Select The Company Names")

        for i, buyer_name in enumerate(test_df['Buyer_Name_Lower']):
            if buyer_name in master_df['Master_Name_Lower'].values:
                # Exact match
                matched_row = master_df.loc[master_df['Master_Name_Lower'] == buyer_name].iloc[0]
                buyer_codes.append(matched_row['Master_Code'])
                updated_buyer_names.append(matched_row['Master_Name'])  # Optional: unify names
            else:
                # Fuzzy match
                tfidf_buyer = vectorizer.transform([buyer_name])
                cosine_scores = cosine_similarity(tfidf_buyer, tfidf_master).flatten()
                top_indices = cosine_scores.argsort()[-5:][::-1]
                top_matches = [(master_df.iloc[idx]['Master_Name'], cosine_scores[idx]) for idx in top_indices]

                match_options = [f"{name} (score: {score:.2f})" for name, score in top_matches]

                st.markdown(f"**Buyer Name:** {test_df['Buyer_Name'].iloc[i]}")
                selected = st.selectbox("Select the best match", match_options, key=i)

                selected_name = selected.split(" (score")[0]
                selected_row = master_df[master_df['Master_Name'] == selected_name].iloc[0]

                buyer_codes.append(selected_row['Master_Code'])
                updated_buyer_names.append(selected_row['Master_Name'])  

        # Insert Buyer_Code column before Buyer_Name
        insert_pos = test_df.columns.get_loc('Buyer_Name')
        test_df.insert(insert_pos, 'Buyer_Code', buyer_codes)

        # Update Buyer_Name with selected names
        test_df['Buyer_Name'] = updated_buyer_names

        # Drop helper
        test_df.drop(columns=['Buyer_Name_Lower'], inplace=True)
        master_df.drop(columns=['Master_Name_Lower'], inplace=True)

        st.subheader("Updated Dataset")
        st.dataframe(test_df)

        # Download updated test data
        st.download_button("Download Updated Dataset", test_df.to_csv(index=False), file_name="updated_test_data.csv", mime='text/csv')
