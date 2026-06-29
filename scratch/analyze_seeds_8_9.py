import pandas as pd
import os

dir_ort = r"C:\Users\123\Desktop\李信翰\碩一\PPO_FJSP\FJSP-DRL-main_NO_GNN\plots\global\20260621_005340_odprog"
csv_details_ort = os.path.join(dir_ort, "details_r161_t05097.csv")
df_ort = pd.read_csv(csv_details_ort)

print("=== OR-Tools Job 63 Details ===")
print(df_ort[df_ort['job'] == 63].sort_values(by='op')[['op', 'machine', 'start', 'end', 'duration', 'status']])

print("\n=== OR-Tools Job 0 Details ===")
print(df_ort[df_ort['job'] == 0].sort_values(by='op')[['op', 'machine', 'start', 'end', 'duration', 'status']])

# Let's print some other job, say Job 5, to see if start/end times are also huge.
print("\n=== OR-Tools Job 5 Details ===")
print(df_ort[df_ort['job'] == 5].sort_values(by='op')[['op', 'machine', 'start', 'end', 'duration', 'status']])
