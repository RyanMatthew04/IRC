import streamlit as st
import pandas as pd
from rapidfuzz.distance.JaroWinkler import normalized_similarity
from itertools import permutations

st.title("Company Name Matching")


# Upload files
master_file = st.file_uploader("Upload Master Dataset", type=["csv", "xlsx"], key="master")
test_file = st.file_uploader("Upload Test Dataset", type=["csv", "xlsx"], key="test")


if master_file and test_file:
    def load_file(file):
        return pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

    master_df = load_file(master_file)
    test_df = load_file(test_file)

    # Validate required columns
    if 'Master_Code' not in master_df.columns or 'Master_Name' not in master_df.columns:
        st.error("Master dataset must contain 'Master_Code' and 'Master_Name' columns.")
    elif 'Buyer_Name' not in test_df.columns:
        st.error("Test dataset must contain 'Buyer_Name' column.")
    else:
        st.success("Both files loaded successfully.")

        # Normalize names
        master_df['Master_Name_Clean'] = master_df['Master_Name'].str.lower().apply(str.strip)
        test_df['Buyer_Name_Clean'] = test_df['Buyer_Name'].str.lower().apply(str.strip)


        # Prepare outputs
        buyer_codes = []
        updated_buyer_names = []

        # Load SBERT model only if needed
        model = None

        st.subheader("Indian Company Name Matching")

        for i, test_name_clean in enumerate(test_df['Buyer_Name_Clean']):
            original_name = test_df['Buyer_Name'].iloc[i]

            # Try exact match
            exact_match = master_df[master_df['Master_Name_Clean'] == test_name_clean]

            if not exact_match.empty:
                # Use the first exact match found
                matched_row = exact_match.iloc[0]
                buyer_codes.append(matched_row['Master_Code'])
                updated_buyer_names.append(matched_row['Master_Name'])
            
            else:
                def permuted_winkler_distance(a, b):
                    """
                    Computes a distance = 1 - max Jaro-Winkler similarity over all token permutations of `a`.
                    Lower distance = better match.
                    """
                    tokens = a.split()
                    max_sim = 0.0
                    for perm in permutations(tokens):
                        permuted = " ".join(perm)
                        sim = normalized_similarity(permuted, b) / 100  # convert 0–100 to 0–1
                        if sim > max_sim:
                            max_sim = sim
                    return 1.0 - max_sim

                distances = master_df['Master_Name_Clean'].apply(
                    lambda master_clean: permuted_winkler_distance(test_name_clean, master_clean)
                )

                top_indices = distances.nsmallest(10).index
                top_matches = [
                    (master_df.loc[idx, 'Master_Name'], distances[idx]) 
                    for idx in top_indices
                ]

                match_options = [
                    f"{name} (distance: {dist:.3f})" 
                    for name, dist in top_matches
                ]

                st.markdown(f"**Buyer Name:** {original_name}")
                selected = st.selectbox(
                    "Select the best match", 
                    match_options, 
                    key=f"permuted_winkler_{i}"
                )

                selected_name = selected.split(" (distance")[0]
                selected_row = master_df[master_df['Master_Name'] == selected_name].iloc[0]

                buyer_codes.append(selected_row['Master_Code'])
                updated_buyer_names.append(selected_row['Master_Name'])
                
    
        # Insert Buyer_Code column before Buyer_Name
        insert_pos = test_df.columns.get_loc('Buyer_Name')
        test_df.insert(insert_pos, 'Buyer_Code', buyer_codes)

        # Update Buyer_Name with matched names
        test_df['Buyer_Name'] = updated_buyer_names

        # Drop helper columns
        test_df.drop(columns=['Buyer_Name_Clean'], inplace=True)
        master_df.drop(columns=['Master_Name_Clean'], inplace=True)

        st.subheader("Updated Dataset")
        st.dataframe(test_df)

        # Download updated test data
        st.download_button("Download Updated Dataset", test_df.to_csv(index=False), file_name="updated_test_data.csv", mime='text/csv')
