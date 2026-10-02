import pandas as pd
import pulp
from pulp import LpVariable, LpProblem, lpSum, LpStatus
import streamlit as st
#HOW TO RUN -- OFFLINE
# 1. Buka New Terminal
# 2. ketik "streamlit run CLP.py"


st.set_page_config(
    page_title="Optimization Capacitated Warehouse Location",
    layout="wide",
)

st.title(
    "🏭 Optimasi Lokasi Gudang & Distribusi", anchor=False
)

st.markdown(
    "Aplikasi ini menyelesaikan Capacitated Location Problem"
    " Model untuk menentukan gudang yang dibuka serta alokasi pengiriman"
    " dengan biaya minimum."
)

# ---------------------------------------------------------
# 1. INISIALISASI DATA AWAL 
# ---------------------------------------------------------
gudang_default = ["W1", "W2", "W3", "W4"]
pasar_default = ["M1", "M2", "M3", "M4", "M5", "M6"]

# Matriks Biaya awal
cost_data_default = {
    "M1": [141, 300, 522, 680],
    "M2": [50, 180, 354, 510],
    "M3": [250, 50, 283, 430],
    "M4": [354, 212, 50, 206],
    "M5": [502, 304, 141, 158],
    "M6": [618, 472, 224, 71],
}
df_cost_init = pd.DataFrame(cost_data_default, index=gudang_default)

# Parameters Gudang awal
df_warehouse_init = pd.DataFrame(
    {
        "Fixed Cost (Juta Rp)": [700.0, 700.0, 700.0, 700.0],
        "Kapasitas (Juta Botol)": [234.0, 234.0, 234.0, 234.0],
    },
    index=gudang_default,
)

# Demand awal
df_demand_init = pd.DataFrame(
    {"Demand (Juta Botol)": [104.0, 93.6, 130.0, 104.0, 114.4, 89.44]},
    index=pasar_default,
).T

# ---------------------------------------------------------
# 2. INTERACTIVE DATA EDITORS 
# ---------------------------------------------------------
st.subheader("📌 Input Parameter Model", anchor=False)

col1, col2 = st.columns([2, 1])

with col1:
    st.write(
        "**1. Matriks Biaya Transportasi / Variabel ($c_{ij}$)** "
        "*(Klik sel"
        " untuk mengubah nilai)*"
    )
    edited_cost_df = st.data_editor(df_cost_init, use_container_width=True)

    st.write(
        "**2. Permintaan Pasar / Demand ($D_j$)** *(Klik sel untuk mengubah"
        " nilai)*"
    )
    edited_demand_df = st.data_editor(
        df_demand_init, use_container_width=True
    )

with col2:
    st.write(
        "**3. Parameter Gudang ($f_i$ & $K_i$)** *(Klik sel untuk mengubah"
        " nilai)*"
    )
    edited_warehouse_df = st.data_editor(
        df_warehouse_init, use_container_width=True
    )

# Extract Data Terupdate dari Editor
gudang = list(edited_cost_df.index)
pasar = list(edited_cost_df.columns)

fixed_cost = edited_warehouse_df["Fixed Cost (Juta Rp)"].to_dict()
capacity = edited_warehouse_df["Kapasitas (Juta Botol)"].to_dict()
demand = edited_demand_df.loc["Demand (Juta Botol)"].to_dict()

# ---------------------------------------------------------
# 3. PEMBUATAN & PENYELESAIAN MODEL OPTIMASI 
# ---------------------------------------------------------
if st.button("🚀 Jalankan Optimasi", type="primary"):
    model = pulp.LpProblem(
        "Capacitated_Warehouse_Location", pulp.LpMinimize
    )

    # Variabel Keputusan: y[i] = biner status gudang (1 = buka, 0 = tutup)
    y = pulp.LpVariable.dicts("Warehouse_Status", gudang, cat="Binary")

    # Variabel Keputusan: x[i, j] = jumlah pengiriman dari gudang i ke pasar j
    x = pulp.LpVariable.dicts(
        "Shipment", [(w, m) for w in gudang for m in pasar], lowBound=0
    )

    # Fungsi Tujuan (Objective Function): Total Cost = Total Fixed + Total Variable
    total_fixed_cost = pulp.lpSum([fixed_cost[w] * y[w] for w in gudang])
    total_variable_cost = pulp.lpSum(
        [
            edited_cost_df.loc[w, m] * x[(w, m)]
            for w in gudang
            for m in pasar
        ]
    )

    model += total_fixed_cost + total_variable_cost, "Total_Cost"

    # Kendala 1: Memenuhi Seluruh Permintaan Pasar (Demand Constraint)
    for m in pasar:
        model += (
            pulp.lpSum([x[(w, m)] for w in gudang]) == demand[m],
            f"Demand_Constraint_{m}",
        )

    # Kendala 2: Tidak Melebihi Kapasitas Gudang (Capacity Constraint)
    for w in gudang:
        model += (
            pulp.lpSum([x[(w, m)] for m in pasar]) <= capacity[w] * y[w],
            f"Capacity_Constraint_{w}",
        )

    # Selesaikan Model
    model.solve()

    # ---------------------------------------------------------
    # 4. HASIL OPTIMASI & VISUALISASI
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Hasil Optimasi", anchor=False)

    status = pulp.LpStatus[model.status]
    if status == "Optimal":
        st.success("✅ Solusi Optimal Ditemukan!")

        # Metric Utama
        st.metric(
            label="💰 Total Biaya Minimum (Juta)",
            value=f"Rp {pulp.value(model.objective):,.2f}",
        )

        col_res1, col_res2 = st.columns(2)

        with col_res1:
            st.write("**1. Status Pembukaan Gudang ($Y_i$)**")
            status_data = []
            for w in gudang:
                is_open = int(y[w].varValue)
                total_sent = sum(x[(w, m)].varValue for m in pasar)
                cap = capacity[w]
                utilization = (total_sent / cap) * 100 if is_open else 0
                status_data.append(
                    {
                        "Gudang": w,
                        "Status": "DIBUKA" if is_open == 1 else "DITUTUP",
                        "Total Pengiriman": total_sent,
                        "Kapasitas Maksimal": cap,
                        "Utilitas Kapasitas (%)": f"{utilization:.2f}%",
                    }
                )

            st.dataframe(pd.DataFrame(status_data), use_container_width=True)

        with col_res2:
            st.write("**2. Matriks Alokasi Pengiriman ($X_{ij}$)**")
            shipment_matrix = pd.DataFrame(index=gudang, columns=pasar)
            for w in gudang:
                for m in pasar:
                    shipment_matrix.loc[w, m] = x[(w, m)].varValue

            st.dataframe(shipment_matrix, use_container_width=True)

    else:
        st.error(
            f"Solusi tidak ditemukan. Status: {status}. Coba periksa apakah"
            " total kapasitas gudang yang dibuka mencukupi total demand."
        )