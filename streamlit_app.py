import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_excel("DATASET 211024.xlsx", sheet_name="BENER")
    df = df.dropna(subset=[
        "Provinsi", "Tahun", "Kesehatan + Pendidikan",
        "Output Layanan Kesehatan", "AHH", "Unmeet Need"
    ])
    df["ln_blj_kesehatan_pendidikan"] = np.log(df["Kesehatan + Pendidikan"])
    df["ln_output_layanan_kesehatan"] = np.log(df["Output Layanan Kesehatan"])
    return df

# Regression model (dynamic)
def get_dynamic_coefficients(df):
    X = df[["ln_blj_kesehatan_pendidikan", "ln_output_layanan_kesehatan"]]
    X = sm.add_constant(X)
    model_ahh = sm.OLS(df["AHH"], X).fit()
    model_unmet = sm.OLS(df["Unmeet Need"], X).fit()
    return model_ahh.params, model_unmet.params

df = load_data()
provinces = df["Provinsi"].unique()
years = sorted(df["Tahun"].unique())

st.title("📊 Dashboard BLU: Dampak Belanja terhadap Kesehatan")

st.sidebar.header("🧭 Filter")
selected_provinces = st.sidebar.multiselect("Pilih Provinsi", provinces, default=list(provinces[:3]))
selected_years = st.sidebar.multiselect("Pilih Tahun", years, default=years)
use_dynamic = st.sidebar.checkbox("Gunakan Koefisien Regresi Dinamis", value=True)

filtered_df = df[df["Provinsi"].isin(selected_provinces) & df["Tahun"].isin(selected_years)]

st.subheader("1. Visualisasi Tren dan Korelasi")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Angka Harapan Hidup (AHH)**")
    st.plotly_chart(
        px.line(filtered_df, x="Tahun", y="AHH", color="Provinsi", markers=True),
        use_container_width=True
    )

with col2:
    st.markdown("**Unmet Need**")
    st.plotly_chart(
        px.line(filtered_df, x="Tahun", y="Unmeet Need", color="Provinsi", markers=True),
        use_container_width=True
    )

with col3:
    st.markdown("**Korelasi AHH vs Belanja**")
    scatter = px.scatter(
        filtered_df, x="ln_blj_kesehatan_pendidikan", y="AHH",
        color="Provinsi", size="Output Layanan Kesehatan", 
        trendline="ols", title=""
    )
    st.plotly_chart(scatter, use_container_width=True)

# Simulation Panel
st.subheader("2. 🎛️ Simulasi Dampak Belanja dan Output")
st.markdown("Atur nilai input untuk mensimulasikan AHH dan Unmet Need berdasarkan model regresi:")

ln_spending = st.slider("Log Belanja Kesehatan + Pendidikan", 22.0, 29.0, 26.5, 0.1)
ln_output = st.slider("Log Output Layanan Kesehatan", 9.0, 13.5, 11.0, 0.1)

if use_dynamic:
    ahh_coef, unmet_coef = get_dynamic_coefficients(df)
    pred_ahh = ahh_coef["const"] + ahh_coef["ln_blj_kesehatan_pendidikan"] * ln_spending + ahh_coef["ln_output_layanan_kesehatan"] * ln_output
    pred_unmet = unmet_coef["const"] + unmet_coef["ln_blj_kesehatan_pendidikan"] * ln_spending + unmet_coef["ln_output_layanan_kesehatan"] * ln_output
else:
    pred_ahh = 4.105 + 0.00183 * ln_spending + 0.00884 * ln_output
    pred_unmet = 1.870 + 0.0222 * ln_spending - 0.0721 * ln_output

st.metric("📈 Prediksi AHH", f"{pred_ahh:.2f}")
st.metric("📉 Prediksi Unmet Need", f"{pred_unmet:.2f}")

st.divider()
st.caption("Dikembangkan untuk Direktorat PPKBLU DJPb – Pemantauan Dampak Belanja BLU Berbasis Data")
