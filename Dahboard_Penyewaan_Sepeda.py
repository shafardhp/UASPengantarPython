import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Konfigurasi tampilan
st.set_page_config(page_title="Dashboard Penyewaan Sepeda",
                   layout="wide",
                   page_icon="🚲")

# Load data
day = pd.read_csv("day.csv")
hour = pd.read_csv("hour.csv")
day["dteday"] = pd.to_datetime(day["dteday"])
hour["dteday"] = pd.to_datetime(hour["dteday"])

# Sidebar filters
st.sidebar.header("Filter Data")

# Date input dengan penanganan yang benar
date_range = st.sidebar.date_input(
    "Rentang Tanggal",
    value=(day["dteday"].min(), day["dteday"].max()),
    min_value=day["dteday"].min(),
    max_value=day["dteday"].max()
)

# Penanganan tanggal yang lebih robust
if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = date_range[0], date_range[0]
    st.sidebar.warning("Pilih rentang tanggal yang valid")

weather_options = st.sidebar.multiselect(
    "Kondisi Cuaca", [1, 2, 3], default=[1, 2, 3],
    format_func=lambda x: {1: "Cerah", 2: "Mendung", 3: "Hujan"}[x]
)

season_options = st.sidebar.multiselect(
    "Musim", [1, 2, 3, 4], default=[1, 2, 3, 4],
    format_func=lambda x: {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}[x]
)

# Filter data
filtered_day = day[
    (day["dteday"] >= pd.to_datetime(start_date)) &
    (day["dteday"] <= pd.to_datetime(end_date)) &
    (day["weathersit"].isin(weather_options)) &
    (day["season"].isin(season_options))
]

# Judul
st.title("Analisis Tren dan Pola Penyewaan Sepeda Berdasarkan Faktor Waktu dan Cuaca pada Sistem Capital Bike Share (2011–2012)")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 EDA", "🧠 Clustering", "⏰ Tren Waktu"])

with tab1:
    st.markdown("## 📋 Data Mentah dan Statistik")
    with st.expander("📎 Kredit"):
        st.write("**Kelompok 3**")
        st.write("- Dinastisya Vasha Agysta (M0722034)")
        st.write("- Mayisya Najmuts Zahra A (M0722048)")
        st.write("- Shafa Ardhana Putri S (M0722072)")
        st.write("Sumber data: Bike-sharing Dataset")

    with st.expander("🗃️ Tabel Data (head)"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Harian")
            st.dataframe(day.head())
        with col2:
            st.subheader("Data Per Jam")
            st.dataframe(hour.head())

    st.markdown("### 📊 Statistik Deskriptif")
    st.dataframe(filtered_day.describe())

    st.markdown("## 📈 Visualisasi Eksploratif")

    # Rata-rata penyewaan per musim
    season_map = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
    filtered_day["season_str"] = filtered_day["season"].map(season_map)
    season_avg = filtered_day.groupby("season_str")["cnt"].mean().reset_index()

    fig1, ax1 = plt.subplots()
    sns.barplot(x="season_str", y="cnt", data=season_avg, ax=ax1)
    ax1.set_title("Rata-rata Penyewaan per Musim")
    st.pyplot(fig1)
    plt.close(fig1)

    # Suhu vs jumlah penyewaan
    fig2, ax2 = plt.subplots()
    sns.regplot(x="temp", y="cnt", data=filtered_day, ax=ax2, scatter_kws={"alpha": 0.4})
    ax2.set_title("Suhu vs Jumlah Penyewaan")
    st.pyplot(fig2)
    plt.close(fig2)

    # Pie chart cuaca
    weather_map = {1: "Cerah", 2: "Mendung", 3: "Hujan"}
    weather_avg = filtered_day.groupby("weathersit")["cnt"].mean().reset_index()
    weather_avg["label"] = weather_avg["weathersit"].map(weather_map)

    fig3, ax3 = plt.subplots()
    ax3.pie(weather_avg["cnt"], labels=weather_avg["label"], autopct="%1.1f%%", startangle=90)
    ax3.set_title("Distribusi Penyewaan Berdasarkan Cuaca")
    st.pyplot(fig3)
    plt.close(fig3)

    # Scatter suhu vs kelembaban
    fig4, ax4 = plt.subplots()
    sns.scatterplot(x="temp", y="hum", hue="cnt", data=filtered_day, palette="coolwarm", ax=ax4)
    ax4.set_title("Suhu vs Kelembaban (+Jumlah Penyewaan)")
    st.pyplot(fig4)
    plt.close(fig4)

with tab2:
    st.markdown("## 🧠 Clustering")

    # Clustering
    features = filtered_day[["temp", "hum", "windspeed", "cnt"]]
    if not features.empty:
        scaled = StandardScaler().fit_transform(features)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
        filtered_day["cluster"] = kmeans.fit_predict(scaled)
        cluster_map = {0: "Low Demand", 1: "Medium Demand", 2: "High Demand"}
        filtered_day["Cluster"] = filtered_day["cluster"].map(cluster_map)

        # Scatter suhu vs cnt
        fig5, ax5 = plt.subplots()
        sns.scatterplot(x="temp", y="cnt", hue="Cluster", data=filtered_day, palette="Set2", ax=ax5)
        ax5.set_title("Clustering Suhu vs Jumlah Penyewaan")
        st.pyplot(fig5)
        plt.close(fig5)

        # Countplot cluster
        fig6, ax6 = plt.subplots()
        sns.countplot(x="Cluster", data=filtered_day, order=["Low Demand", "Medium Demand", "High Demand"], ax=ax6)
        ax6.set_title("Distribusi Kategori Penyewaan")
        st.pyplot(fig6)
        plt.close(fig6)

        # Boxplot suhu per cluster
        fig7, ax7 = plt.subplots()
        sns.boxplot(x="Cluster", y="temp", data=filtered_day, 
                   order=["Low Demand", "Medium Demand", "High Demand"], ax=ax7)
        ax7.set_title("Distribusi Suhu per Kategori")
        st.pyplot(fig7)
        plt.close(fig7)

        # Scatter kelembaban vs cnt
        fig8, ax8 = plt.subplots()
        sns.scatterplot(x="hum", y="cnt", hue="Cluster", data=filtered_day, palette="Set2", ax=ax8)
        ax8.set_title("Kelembaban vs Jumlah Penyewaan")
        st.pyplot(fig8)
        plt.close(fig8)
    else:
        st.warning("Tidak ada data yang tersedia untuk clustering dengan filter yang dipilih.")

with tab3:
    st.markdown("## ⏰ Pola Penyewaan Berdasarkan Waktu")

    # Rata-rata per hari dalam seminggu
    weekday_map = {
        0: "Minggu", 1: "Senin", 2: "Selasa", 3: "Rabu",
        4: "Kamis", 5: "Jumat", 6: "Sabtu"
    }
    weekday_avg = filtered_day.groupby("weekday")["cnt"].mean().reset_index()
    weekday_avg["day"] = weekday_avg["weekday"].map(weekday_map)

    fig9, ax9 = plt.subplots()
    sns.barplot(x="day", y="cnt", data=weekday_avg, color="skyblue", ax=ax9)
    sns.lineplot(x="day", y="cnt", data=weekday_avg, marker="o", color="blue", ax=ax9)
    ax9.set_title("Rata-rata Per Hari (Minggu - Sabtu)")
    st.pyplot(fig9)
    plt.close(fig9)

    # Rata-rata per bulan
    month_map = {i: name for i, name in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"], start=1)}
    month_avg = filtered_day.groupby("mnth")["cnt"].mean().reset_index()
    month_avg["month"] = month_avg["mnth"].map(month_map)

    fig10, ax10 = plt.subplots()
    sns.barplot(x="month", y="cnt", data=month_avg, color="lightgreen", ax=ax10)
    sns.lineplot(x="month", y="cnt", data=month_avg, marker="o", color="green", ax=ax10)
    ax10.set_title("Rata-rata Per Bulan (Jan - Des)")
    st.pyplot(fig10)
    plt.close(fig10)

    # Data hourly yang difilter
    hour_filtered = hour[
        (hour["dteday"] >= pd.to_datetime(start_date)) &
        (hour["dteday"] <= pd.to_datetime(end_date)) &
        (hour["weathersit"].isin(weather_options))
    ]

    # Rata-rata penyewaan per jam
    if not hour_filtered.empty:
        hour_avg = hour_filtered.groupby("hr")["cnt"].mean().reset_index()
        fig11, ax11 = plt.subplots()
        sns.lineplot(x="hr", y="cnt", data=hour_avg, marker="o", ax=ax11)
        ax11.set_title("Rata-rata Penyewaan per Jam")
        st.pyplot(fig11)
        plt.close(fig11)

        # Heatmap jam × hari
        hour_filtered["weekday"] = hour_filtered["weekday"].map(weekday_map)
        heatmap_data = hour_filtered.pivot_table(index="hr", columns="weekday", values="cnt", aggfunc="mean")

        fig12, ax12 = plt.subplots(figsize=(10, 4))
        sns.heatmap(heatmap_data, cmap="YlGnBu", ax=ax12)
        ax12.set_title("Heatmap Jam × Hari")
        st.pyplot(fig12)
        plt.close(fig12)
    else:
        st.warning("Tidak ada data hourly yang tersedia untuk rentang filter yang dipilih.")