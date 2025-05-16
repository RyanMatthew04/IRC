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
                from itertools import permutations
                import re

                # Function to clean company suffixes (for Jaccard only)
                def clean_company_name_for_jaccard(name):
                    suffixes = r"\b(incorporated|inc|llc|ltd|limited|corp|corporation|plc|co|company|pvt|private)\b"
                    name = name.lower()
                    name = re.sub(suffixes, '', name)
                    name = re.sub(r'\s+', ' ', name)  
                    return name.strip()

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

                def jaccard_distance(a, b):
                    """
                    Computes Jaccard distance after cleaning common company suffixes.
                    """
                    a_clean = clean_company_name_for_jaccard(a)
                    b_clean = clean_company_name_for_jaccard(b)

                    set_a = set(a_clean.split())
                    set_b = set(b_clean.split())
                    intersection = set_a & set_b
                    union = set_a | set_b
                    if not union:
                        return 1.0
                    return 1.0 - len(intersection) / len(union)

                # Compute distances using both methods
                winkler_distances = master_df['Master_Name_Clean'].apply(
                    lambda master_clean: permuted_winkler_distance(test_name_clean, master_clean)
                )
                jaccard_distances = master_df['Master_Name_Clean'].apply(
                    lambda master_clean: jaccard_distance(test_name_clean, master_clean)
                )

                # Get top 5 from each method
                top_winkler_indices = winkler_distances.nsmallest(10).index
                top_jaccard_indices = jaccard_distances.nsmallest(10).index

                # Combine results with method label
                top_matches = []

                # Zip the two lists and interleave them
                for winkler_idx, jaccard_idx in zip(top_winkler_indices, top_jaccard_indices):
                    top_matches.append((
                        master_df.loc[winkler_idx, 'Master_Name'],
                        winkler_distances[winkler_idx],
                        'Permuted Winkler'
                    ))
                    top_matches.append((
                        master_df.loc[jaccard_idx, 'Master_Name'],
                        jaccard_distances[jaccard_idx],
                        'Jaccard'
                    ))

                # If one list is longer, add remaining entries
                longer_winkler = top_winkler_indices[len(top_jaccard_indices):]
                longer_jaccard = top_jaccard_indices[len(top_winkler_indices):]

                for idx in longer_winkler:
                    top_matches.append((
                        master_df.loc[idx, 'Master_Name'],
                        winkler_distances[idx],
                        'Permuted Winkler'
                    ))

                for idx in longer_jaccard:
                    top_matches.append((
                        master_df.loc[idx, 'Master_Name'],
                        jaccard_distances[idx],
                        'Jaccard'
                    ))


                unique_matches = {}
                for name, dist, method in top_matches:
                    if name not in unique_matches:
                        unique_matches[name] = (dist, method)

                # Convert back to list of tuples
                deduped_matches = [(name, dist, method) for name, (dist, method) in unique_matches.items()][:10]


                # Create dropdown options
                match_options = [
                    f"{name}"
                    for name, dist, method in deduped_matches
                ]


                st.markdown(f"**Buyer Name:** {original_name}")
                selected = st.selectbox(
                    "Select the best match", 
                    match_options, 
                    key=f"combined_match_{i}"
                )

                # Extract selected name
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
