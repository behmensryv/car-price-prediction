import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from sklearn.linear_model import LinearRegression


# =========================
# 1. Səhifə ayarları
# =========================

st.set_page_config(
    page_title="Avtomobil Qiymət Analizi",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Avtomobil Qiymət Analizi")


# =========================
# 2. Database-dən məlumatı oxu
# =========================

conn = sqlite3.connect("avtomobil.db")

df = pd.read_sql_query(
    "SELECT * FROM avtomobil_temiz",
    conn
)

conn.close()


# =========================
# 3. Filtrlər
# =========================

st.sidebar.header("Filtrlər")

# Marka filteri
markalar = ["Hamısı"] + sorted(df["Marka"].dropna().unique().tolist())

secili_marka = st.sidebar.selectbox(
    "Marka",
    markalar
)


# Buraxılış ili filteri
min_il = int(df["Buraxılış_ili"].min())
max_il = int(df["Buraxılış_ili"].max())

il_araligi = st.sidebar.slider(
    "Buraxılış ili",
    min_value=min_il,
    max_value=max_il,
    value=(min_il, max_il)
)


# =========================
# 4. Filtrləmə
# =========================

df_filtered = df.copy()

if secili_marka != "Hamısı":
    df_filtered = df_filtered[
        df_filtered["Marka"] == secili_marka
    ]

df_filtered = df_filtered[
    (df_filtered["Buraxılış_ili"] >= il_araligi[0]) &
    (df_filtered["Buraxılış_ili"] <= il_araligi[1])
]


# =========================
# 5. Cədvəl
# =========================

st.subheader("Filtrə uyğun avtomobillər")

st.write(
    f"Tapılan avtomobillərin sayı: **{len(df_filtered)}**"
)

st.dataframe(
    df_filtered,
    use_container_width=True
)


# =========================
# 6. Qrafik
# =========================

st.subheader("Markalara görə orta qiymət")

if len(df_filtered) > 0:

    chart_data = (
        df_filtered
        .groupby("Marka")["Qiymət_AZN"]
        .mean()
        .reset_index()
        .sort_values("Qiymət_AZN", ascending=False)
    )

    fig = px.bar(
        chart_data,
        x="Marka",
        y="Qiymət_AZN",
        title="Markalara görə orta avtomobil qiyməti",
        labels={
            "Marka": "Marka",
            "Qiymət_AZN": "Orta qiymət (AZN)"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:
    st.warning("Seçilmiş filtrə uyğun avtomobil tapılmadı.")


# =========================
# 7. Qiymət proqnozu
# =========================

st.subheader("💰 Avtomobil qiymətinin proqnozu")

st.write(
    "Avtomobil haqqında məlumatları daxil edin:"
)


# Model üçün istifadə ediləcək sütunlar
features = [
    "Buraxılış_ili",
    "Mühərrik_Həcmi",
    "Yürüş_KM"
]

target = "Qiymət_AZN"


# =========================
# 8. NaN problemini həll et
# =========================

df_model = df[features + [target]].dropna()


X = df_model[features]
y = df_model[target]


# =========================
# 9. Linear Regression modeli
# =========================

model = LinearRegression()

model.fit(X, y)


# =========================
# 10. User inputları
# =========================

col1, col2, col3 = st.columns(3)

with col1:
    daxil_il = st.number_input(
        "Buraxılış ili",
        min_value=min_il,
        max_value=max_il,
        value=max_il,
        step=1
    )

with col2:
    daxil_muherrik = st.number_input(
        "Mühərrik həcmi",
        min_value=0.1,
        max_value=10.0,
        value=2.0,
        step=0.1
    )

with col3:
    daxil_yurus = st.number_input(
        "Yürüş (KM)",
        min_value=0,
        max_value=1000000,
        value=100000,
        step=1000
    )


# =========================
# 11. Proqnoz
# =========================

if st.button("Qiyməti proqnozlaşdır"):

    yeni_avtomobil = pd.DataFrame({
        "Buraxılış_ili": [daxil_il],
        "Mühərrik_Həcmi": [daxil_muherrik],
        "Yürüş_KM": [daxil_yurus]
    })

    proqnoz = model.predict(yeni_avtomobil)[0]

    st.success(
        f"🚗 Təxmini avtomobil qiyməti: **{proqnoz:,.0f} AZN**"
    )