import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------
# 1. SƏHİFƏ AYARLARI VƏ DİZAYN (CSS)
# ---------------------------------------------------------
st.set_page_config(
    page_title="AutoMarket - Tam Analitika və Qiymətləndirmə",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        background-color: #f1f3f5;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: white !important;
    }
    .price-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #38bdf8;
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        margin-top: 10px;
        border: 1px solid #334155;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. MƏLUMATLARIN YÜKLƏNMƏSİ VƏ MODELİN TƏLİMİ
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("avtomobil_temiz.csv")
    # Əgər 'Yaş' sütunu daxil edilməyibsə, avtomatik hesabla
    current_year = datetime.now().year
    if 'Yaş' not in df.columns or df['Yaş'].isnull().any():
        df['Yaş'] = current_year - df['Buraxılış_ili']
    return df

@st.cache_resource
def train_model(df):
    # ML modeli üçün BÜTÜN xüsusiyyətlər
    features = ['Marka', 'Model', 'Buraxılış_ili', 'Yaş', 'Mühərrik_Həcmi', 'Yürüş_KM', 
                'Yanacaq', 'Sürətlər_Qutusu', 'Ban_novu', 'Rəng', 'Şəhər', 
                'Vuruq', 'Rənglənib', 'Sahib_sayi']
    target = 'Qiymət_AZN'

    model_df = df[features + [target]].dropna(subset=[target])

    X = model_df[features]
    y = model_df[target]

    cat_cols = ['Marka', 'Model', 'Yanacaq', 'Sürətlər_Qutusu', 'Ban_novu', 'Rəng', 'Şəhər', 'Vuruq', 'Rənglənib', 'Sahib_sayi']
    num_cols = ['Buraxılış_ili', 'Yaş', 'Mühərrik_Həcmi', 'Yürüş_KM']

    num_transformer = SimpleImputer(strategy='median')
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    model.fit(X, y)
    return model

df = load_data()
model = train_model(df)

# ---------------------------------------------------------
# 3. YAN PANEL (SIDEBAR) — FİLTRLƏR
# ---------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/744/741407.png", width=80)
st.sidebar.title("🚗 AutoMarket Control")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio("Bölməni seçin:", ["📊 Analitika & Filtrləmə", "🔮 Qiymət Təxmini (ML)"])
st.sidebar.markdown("---")

if app_mode == "📊 Analitika & Filtrləmə":
    st.sidebar.header("🔍 Genişləndirilmiş Filtrlər")
    
    # 1. Marka filtri
    all_brands = ["Hamısı"] + sorted(df['Marka'].dropna().astype(str).unique().tolist())
    selected_brand = st.sidebar.selectbox("Marka", all_brands)
    

    # 6. Qiymət Kateqoriyası filtri
    all_categories = ["Hamısı"] + sorted(df['Qiymət_Kateqoriyası'].dropna().astype(str).unique().tolist())
    selected_category = st.sidebar.selectbox("Qiymət Kateqoriyası", all_categories)

    # 7. İl aralığı filtri
    min_year = int(df['Buraxılış_ili'].min())
    max_year = int(df['Buraxılış_ili'].max())
    selected_year_range = st.sidebar.slider("Buraxılış İli Aralığı", min_value=min_year, max_value=max_year, value=(min_year, max_year))

    # Filtrləmə Məntiqi
    filtered_df = df.copy()
    if selected_brand != "Hamısı":
        filtered_df = filtered_df[filtered_df['Marka'] == selected_brand]
    if selected_category != "Hamısı":
        filtered_df = filtered_df[filtered_df['Qiymət_Kateqoriyası'] == selected_category]
    
    filtered_df = filtered_df[
        (filtered_df['Buraxılış_ili'] >= selected_year_range[0]) & 
        (filtered_df['Buraxılış_ili'] <= selected_year_range[1])
    ]

# ---------------------------------------------------------
# 4. ƏSAS MƏZMUN (MAIN CONTENT)
# ---------------------------------------------------------

st.title("🚙 Avtomobil Bazarı Analitikası və Qiymət Təxmini")
st.caption("Bütün sütunlar və parametrlər üzrə tam analiz və Machine Learning təxmini")
st.markdown("---")

if app_mode == "📊 Analitika & Filtrləmə":

    # KPI METRİKALARI
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Elan Sayı", f"{len(filtered_df)} ədəd")
    m2.metric("Orta Qiymət", f"{filtered_df['Qiymət_AZN'].mean():,.0f} AZN" if not filtered_df.empty else "N/A")
    m3.metric("Orta Avtomobil Yaşı", f"{filtered_df['Yaş'].mean():.1f} il" if not filtered_df.empty else "N/A")
    m4.metric("Orta Yürüş", f"{filtered_df['Yürüş_KM'].mean():,.0f} km" if not filtered_df.empty else "N/A")
    m5.metric("Ən Geniş Rəng", filtered_df['Rəng'].mode()[0] if not filtered_df.empty and not filtered_df['Rəng'].dropna().empty else "N/A")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📋 Bütün Sütunlar daxil Cədvəl", "📈 Vizual Qrafiklər"])

    with tab1:
        st.subheader("Filtrə Uyğun Avtomobil Siyahısı")
        st.dataframe(filtered_df, use_container_width=True, height=450)

    with tab2:
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.markdown("##### 🏎️ Avtomobilin Yaşı və Qiymət Əlaqəsi (Vuruq Vəziyyətinə Göre)")
            if not filtered_df.empty:
                fig_scatter = px.scatter(
                    filtered_df, 
                    x="Yaş", 
                    y="Qiymət_AZN", 
                    color="Vuruq", 
                    hover_data=['Marka', 'Model', 'Rəng', 'Rənglənib'],
                    title="Avtomobil Yaşı vs Qiymət (AZN)",
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                fig_scatter.update_layout(template="plotly_white")
                st.plotly_chart(fig_scatter, use_container_width=True)

        with col_g2:
            st.markdown("##### 🎨 Rənglərə Görə Avtomobil Sayı və Orta Qiymət")
            if not filtered_df.empty:
                color_df = filtered_df.groupby('Rəng').agg({'Qiymət_AZN': 'mean', 'Elan_id': 'count'}).reset_index()
                fig_bar = px.bar(
                    color_df, 
                    x='Rəng', 
                    y='Qiymət_AZN', 
                    color='Rəng',
                    text='Elan_id',
                    title="Rənglər üzrə Orta Qiymət və Elan Sayı"
                )
                fig_bar.update_layout(template="plotly_white", showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

elif app_mode == "🔮 Qiymət Təxmini (ML)":
    st.subheader("⚙️ Avtomobilin Bütün Parametrlərini Daxil Edin")

    current_year = datetime.now().year

    c1, c2, c3 = st.columns(3)

    with c1:
        p_marka = st.selectbox("Marka", sorted(df['Marka'].dropna().unique()))
        available_models = df[df['Marka'] == p_marka]['Model'].dropna().unique()
        p_model = st.selectbox("Model", sorted(available_models))
        p_il = st.number_input("Buraxılış İli", min_value=1980, max_value=current_year, value=2018)
        p_yas = current_year - p_il
        st.info(f"💡 Avtomobilin təxmini Yaşı: **{p_yas} il**")

    with c2:
        p_muherrik = st.number_input("Mühərrik Həcmi (L)", min_value=0.0, max_value=8.0, value=2.0, step=0.1)
        p_yurus = st.number_input("Yürüş (KM)", min_value=0, max_value=1000000, value=120000, step=5000)
        p_yanacaq = st.selectbox("Yanacaq Növü", df['Yanacaq'].dropna().unique())
        p_korobka = st.selectbox("Sürətlər Qutusu", df['Sürətlər_Qutusu'].dropna().unique())

    with c3:
        p_ban = st.selectbox("Ban Növü", df['Ban_novu'].dropna().unique())
        p_reng = st.selectbox("Rəng", df['Rəng'].dropna().unique())
        p_seher = st.selectbox("Şəhər", df['Şəhər'].dropna().unique())
        p_sahib = st.selectbox("Sahib Sayı", df['Sahib_sayi'].dropna().unique())

    st.markdown("---")
    col_sub1, col_sub2 = st.columns(2)
    with col_sub1:
        p_vuruq = st.radio("Vuruğu var?", df['Vuruq'].dropna().unique(), horizontal=True)
    with col_sub2:
        p_renglenib = st.radio("Rənglənib?", df['Rənglənib'].dropna().unique(), horizontal=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔮 Qiyməti Təxmin Et", use_container_width=True, type="primary"):
        # Model üçün tam input DataFrame
        input_data = pd.DataFrame([{
            'Marka': p_marka,
            'Model': p_model,
            'Buraxılış_ili': p_il,
            'Yaş': p_yas,
            'Mühərrik_Həcmi': p_muherrik,
            'Yürüş_KM': p_yurus,
            'Yanacaq': p_yanacaq,
            'Sürətlər_Qutusu': p_korobka,
            'Ban_novu': p_ban,
            'Rəng': p_reng,
            'Şəhər': p_seher,
            'Vuruq': p_vuruq,
            'Rənglənib': p_renglenib,
            'Sahib_sayi': p_sahib
        }])

        pred_price = model.predict(input_data)[0]

        st.markdown(f"""
            <div class="price-box">
                <h3 style="margin:0; color: #94a3b8;">Təxmini Bazar Qiyməti</h3>
                <h1 style="font-size: 45px; margin: 10px 0; color: #38bdf8;">{pred_price:,.0f} AZN</h1>
                <p style="margin:0; font-size: 15px; color: #cbd5e1;">Bütün parametrlər nəzərə alınmaqla ehtimal olunan qiymət dəhlizi: <b>{pred_price*0.93:,.0f} - {pred_price*1.07:,.0f} AZN</b></p>
            </div>
        """, unsafe_allow_html=True)