import pandas as pd


df_state = pd.read_csv("fips-state.csv", dtype="str")

df_county = pd.read_csv("fips-code.csv", dtype="str").assign(
    state_id=lambda df: df.id.str[:2]
)

df_fips = pd.merge(
    df_county, df_state, left_on="state_id", right_on="id", how="inner"
).assign(
    state_id=lambda df: df["state"].apply(lambda x: x.lower()),
    name=lambda df: df["name"].apply(lambda x: x.lower()),
)

df_counties = pd.read_csv("counties.csv", dtype="str").assign(
    state=lambda df: df["state"].apply(lambda x: x.lower()),
    city=lambda df: df["city"].apply(lambda x: x.lower()),
)

df_final = pd.merge(
    df_counties,
    df_fips,
    left_on=["state", "city"],
    right_on=["state_id", "name"],
    how="left",
).rename(
    columns={
        "id_x": "fips",
        "id_y": "fips_state",
    }
)

df_final.to_excel("counties_fips.xlsx", index=False)
