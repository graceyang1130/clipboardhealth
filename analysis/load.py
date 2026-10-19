import sys, pandas as pd
D = sys.argv[1]; S = sys.argv[2]
for name in ["Cleveland_shifts_logs","Booking_logs","Cancel_logs"]:
    df = pd.read_excel(f"{D}/{name}.xlsx", engine="openpyxl")
    df.to_pickle(f"{S}/{name}.pkl")
    print("=====", name, df.shape)
    print(df.dtypes.to_string())
    print("nulls:\n", df.isna().sum().to_string())
