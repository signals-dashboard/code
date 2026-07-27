import pandas as pd

# 1. Load the old and new datasets
old_df = pd.read_csv('data/processed/old_signals.csv')
new_df = pd.read_csv('data/processed/processed_signals.csv')

# 2. Extract the exact column list from the old dataset (our master template)
master_columns = old_df.columns

# 3. Force the new dataset to match the old dataset's structure
# This automatically drops any new/unnecessary columns like "sub_channel_name".
# If the new data is missing an old column, it safely fills it with blank spaces instead of crashing.
new_df_filtered = new_df.reindex(columns=master_columns)

# 4. Stack the old data and the filtered new data together
merged_df = pd.concat([old_df, new_df_filtered], ignore_index=True)

# 5. Remove duplicate rows based on unique signal ID. Retains original record with keep='first'
merged_df = merged_df.drop_duplicates(subset=['signal_id'], keep='first')

# 6. Save the combined data back to the main file, overwriting the buggy one
merged_df.to_csv('data/processed/processed_signals.csv', index=False)

print("Merge complete! The main CSV now has all signals with the original clean structure.")