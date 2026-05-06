import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chemical Testing Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    .stMetric {
        background: white;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .stMetric label { color: #6b7280 !important; font-size: 13px !important; }
    .stMetric .css-1wivap2 { font-size: 28px !important; font-weight: 700 !important; color: #1e3a5f !important; }
    h1 { color: #1e3a5f !important; }
    h2, h3 { color: #1e3a5f !important; }
    .kit-card {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        color: white;
        border-radius: 14px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(30,58,95,0.25);
    }
    .kit-card h3 { color: white !important; margin: 0 0 8px 0; font-size: 18px; }
    .kit-card p { margin: 0; font-size: 13px; opacity: 0.85; }
    .kit-card .count { font-size: 36px; font-weight: 800; margin: 10px 0 4px; }
    .kit-card-green {
        background: linear-gradient(135deg, #065f46, #059669);
    }
    .date-badge {
        background: white;
        border-left: 5px solid #2d6a9f;
        border-radius: 10px;
        padding: 14px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 8px;
    }
    .date-badge .label { font-size: 12px; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .date-badge .value { font-size: 20px; font-weight: 700; color: #1e3a5f; }
    div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
    .section-divider {
        border: none;
        border-top: 2px solid #e5e7eb;
        margin: 10px 0 20px 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── File Uploader ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/test-tube.png", width=60)
    st.markdown("## 📂 อัปโหลดไฟล์ข้อมูล")
    uploaded_file = st.file_uploader(
        "ลาก & วางไฟล์ CSV ที่นี่",
        type=["csv"],
        help="รองรับไฟล์ Database_chemical.csv และไฟล์ที่มีโครงสร้างเดียวกัน",
    )
    if uploaded_file:
        st.success(f"✅ โหลดไฟล์สำเร็จ: {uploaded_file.name}")
    else:
        st.info("⬆️ กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มต้น")
        st.stop()

# ─── Load & Process Data ───────────────────────────────────────────────────────
@st.cache_data
def load_data(file_bytes, file_name):
    import io
    df = pd.read_csv(io.BytesIO(file_bytes))
    # Row 0 is the sub-header row – drop it
    df = df.iloc[1:].reset_index(drop=True)

    # Rename result columns
    df = df.rename(columns={
        "Result":       "Organophosphate",
        "Unnamed: 15":  "Carbamate",
        "Unnamed: 16":  "Organochlorine",
        "Unnamed: 17":  "Pyrethroid",
        "Sample \nCondition": "Sample_Condition",
    })

    # Keep only rows with a sample name
    df = df[df["Sample Name"].notna()].copy()

    # Identify Test Kit:  /05 → GPO TM/2 ;  everything else → GT
    df["lab_suffix"] = df["Lab No."].str.extract(r"/(\d+)$")
    df["Test Kit"] = df["lab_suffix"].apply(
        lambda x: "GPO TM/2 Test Kit" if x == "05" else "GT Test Kit"
    )

    # Parse dates
    df["Received Date"] = pd.to_datetime(df["Received Date"], dayfirst=True, errors="coerce")
    df["Analysis Date"] = pd.to_datetime(df["Analysis Date"], dayfirst=True, errors="coerce")

    # Week label
    iso = df["Received Date"].dt.isocalendar()
    df["week_num"] = iso.week
    df["year"] = iso.year
    df["week_label"] = df.apply(
        lambda r: f"W{int(r['week_num']):02d}" if pd.notna(r["Received Date"]) else None,
        axis=1,
    )

    # Normalise result values: treat NaN as not tested (exclude from counts)
    for col in ["Organophosphate", "Carbamate", "Organochlorine", "Pyrethroid"]:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": None, "-": None, "": None})

    return df


df = load_data(uploaded_file.getvalue(), uploaded_file.name)

# Separate GT and GPO data
gt_df  = df[df["Test Kit"] == "GT Test Kit"].copy()
gpo_df = df[df["Test Kit"] == "GPO TM/2 Test Kit"].copy()

# Valid-dated rows (for weekly chart)
dated_df = df[df["Received Date"].notna()].copy()

# ─── Helper ────────────────────────────────────────────────────────────────────
CHEMICAL_GROUPS = {
    "Organophosphate": ("GT Test Kit",       "#ef4444"),
    "Carbamate":       ("GT Test Kit",       "#f97316"),
    "Organochlorine":  ("GT Test Kit",       "#3b82f6"),
    "Pyrethroid":      ("GT Test Kit",       "#8b5cf6"),
}

def group_stats(data, col):
    valid = data[data[col].notna()]
    total = len(valid)
    detected = int((valid[col] != "Not Detect").sum())
    not_det  = total - detected
    pct_det  = round(detected / total * 100, 1) if total else 0
    pct_nd   = round(not_det  / total * 100, 1) if total else 0
    return total, detected, not_det, pct_det, pct_nd


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🧪 Chemical Residue Testing Dashboard")
st.markdown("**ระบบสรุปผลการตรวจสอบสารเคมีตกค้างในผัก-ผลไม้** · QA Raw Material")
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ─── Section 1 : Test Kits ─────────────────────────────────────────────────────
st.markdown("### 🔬 ชุดทดสอบที่ใช้")
col_gt, col_gpo = st.columns(2)

with col_gt:
    st.markdown(f"""
    <div class="kit-card">
        <h3>GT Test Kit</h3>
        <div class="count">{len(df):,}</div>
        <p>ตัวอย่าง  ·  Organophosphate + Carbamate</p>
    </div>""", unsafe_allow_html=True)

with col_gpo:
    st.markdown(f"""
    <div class="kit-card kit-card-green">
        <h3>GPO TM/2 Test Kit</h3>
        <div class="count">{len(df):,}</div>
        <p>ตัวอย่าง  ·  Organochlorine + Pyrethroid</p>
    </div>""", unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ─── Section 2 : KPI Row ──────────────────────────────────────────────────────
st.markdown("### 📊 สรุปภาพรวม")

total_samples  = len(df)
supplier_count = df["Supplier"].nunique()
sample_types   = df["Sample Name"].nunique()

date_min = dated_df["Received Date"].min()
date_max = dated_df["Received Date"].max()
date_str = (
    f"{date_min.strftime('%d %b %Y')} – {date_max.strftime('%d %b %Y')}"
    if pd.notna(date_min) else "N/A"
)

# Group counts
oc_py_total = len(df[df[["Organochlorine","Pyrethroid"]].notna().any(axis=1)])
op_ca_total = len(df[df[["Organophosphate","Carbamate"]].notna().any(axis=1)])

k1, k2, k3, k4 = st.columns(4)
k1.metric("🧪 ตัวอย่างทั้งหมด",            f"{total_samples:,}")
k2.metric("🏭 จำนวน Supplier",              f"{supplier_count:,}")
k3.metric("🥦 ชนิดตัวอย่าง",                f"{sample_types:,} ชนิด")
k4.metric("📅 ช่วงการทดสอบ",                date_str)

st.markdown("")

g1, g2 = st.columns(2)
g1.metric("🟠 Organophosphate + Carbamate",   f"{op_ca_total:,} ตัวอย่าง")
g2.metric("🔵 Organochlorine + Pyrethroid",   f"{oc_py_total:,} ตัวอย่าง")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ─── Section 3 : Weekly Chart ─────────────────────────────────────────────────
st.markdown("### 📅 จำนวนตัวอย่างรายสัปดาห์ (Detected vs Not Detected)")

# Build weekly data – we combine all 4 cols for detected/not-detected count
def get_weekly_stats(data):
    rows = []
    for _, grp in data.groupby("week_label"):
        week = grp["week_label"].iloc[0]
        yr   = int(grp["year"].iloc[0]) if grp["year"].notna().any() else None
        det_count = 0
        nd_count  = 0
        for col in ["Organophosphate", "Carbamate", "Organochlorine", "Pyrethroid"]:
            valid = grp[grp[col].notna()]
            det_count += int((valid[col] != "Not Detect").sum())
            nd_count  += int((valid[col] == "Not Detect").sum())
        rows.append({"Week": week, "Year": yr, "Detected": det_count, "Not Detected": nd_count})
    return pd.DataFrame(rows).sort_values(["Year","Week"])

weekly_df = get_weekly_stats(dated_df)

fig_weekly = go.Figure()
fig_weekly.add_trace(go.Bar(
    x=weekly_df["Week"], y=weekly_df["Not Detected"],
    name="Not Detected", marker_color="#3b82f6",
    hovertemplate="<b>%{x}</b><br>Not Detected: %{y:,}<extra></extra>",
))
fig_weekly.add_trace(go.Bar(
    x=weekly_df["Week"], y=weekly_df["Detected"],
    name="Detected", marker_color="#ef4444",
    hovertemplate="<b>%{x}</b><br>Detected: %{y:,}<extra></extra>",
))
fig_weekly.update_layout(
    barmode="stack",
    xaxis_title="สัปดาห์", yaxis_title="จำนวนการทดสอบ",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(t=30, b=10),
    height=380,
)
fig_weekly.update_xaxes(tickangle=-35)
st.plotly_chart(fig_weekly, use_container_width=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ─── Section 4 : Supplier & Top 5 Veg in 2 columns ───────────────────────────
col_sup, col_top = st.columns([1, 1])

with col_sup:
    st.markdown("### 🏭 Top Supplier")
    top_sup = (
        df["Supplier"].value_counts()
        .head(10)
        .reset_index()
        .rename(columns={"index": "Supplier", "count": "จำนวนตัวอย่าง"})
    )
    top_sup.columns = ["Supplier", "จำนวนตัวอย่าง"]
    fig_sup = px.bar(
        top_sup.sort_values("จำนวนตัวอย่าง"),
        x="จำนวนตัวอย่าง", y="Supplier",
        orientation="h", text="จำนวนตัวอย่าง",
        color="จำนวนตัวอย่าง",
        color_continuous_scale=["#bfdbfe", "#1d4ed8"],
    )
    fig_sup.update_traces(textposition="outside")
    fig_sup.update_layout(
        showlegend=False, coloraxis_showscale=False,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20, b=10, l=10, r=40),
        height=380,
        xaxis_title="จำนวนตัวอย่าง", yaxis_title="",
    )
    st.plotly_chart(fig_sup, use_container_width=True)

with col_top:
    st.markdown("### 🥦 Top 5 ผักที่ส่งมาทดสอบมากที่สุด")
    top5 = (
        df["Sample Name"].value_counts()
        .head(5)
        .reset_index()
    )
    top5.columns = ["ชื่อตัวอย่าง", "จำนวน"]

    colors = ["#1d4ed8", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd"]
    fig_top5 = go.Figure(go.Bar(
        x=top5["ชื่อตัวอย่าง"],
        y=top5["จำนวน"],
        marker_color=colors,
        text=top5["จำนวน"],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>จำนวน: %{y:,}<extra></extra>",
    ))
    fig_top5.update_layout(
        xaxis_title="ชนิดตัวอย่าง", yaxis_title="จำนวนตัวอย่าง",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20, b=10),
        height=380,
    )
    st.plotly_chart(fig_top5, use_container_width=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ─── Section 5 : Summary Table ────────────────────────────────────────────────
st.markdown("### 📋 ตารางสรุปผลการทดสอบ 4 กลุ่มสารเคมี")

GROUPS = ["Organochlorine", "Pyrethroid", "Organophosphate", "Carbamate"]
GROUP_COLORS = {
    "Organochlorine":  "#dbeafe",
    "Pyrethroid":      "#ede9fe",
    "Organophosphate": "#fee2e2",
    "Carbamate":       "#fef3c7",
}

tabs = st.tabs([f"🔬 {g}" for g in GROUPS])

for tab, group_col in zip(tabs, GROUPS):
    with tab:
        sub = df[df[group_col].notna()].copy()
        if sub.empty:
            st.info("ไม่มีข้อมูลในกลุ่มนี้")
            continue

        # Build summary by Sample Name
        records = []
        for sname, grp in sub.groupby("Sample Name"):
            tot, det, nd, pct_d, pct_nd = group_stats(grp, group_col)
            records.append({
                "Sample Type":   sname,
                "Detected":      det,
                "Not Detected":  nd,
                "%Detected":     pct_d,
                "%Not Detected": pct_nd,
            })

        summary_df = pd.DataFrame(records).sort_values("Detected", ascending=False)

        # Overall totals row
        total_row = pd.DataFrame([{
            "Sample Type":   "✅ รวมทั้งหมด",
            "Detected":      summary_df["Detected"].sum(),
            "Not Detected":  summary_df["Not Detected"].sum(),
            "%Detected":     round(summary_df["Detected"].sum() / len(sub) * 100, 1),
            "%Not Detected": round(summary_df["Not Detected"].sum() / len(sub) * 100, 1),
        }])
        display_df = pd.concat([total_row, summary_df], ignore_index=True)

        col_l, col_r = st.columns([3, 2])

        with col_l:
            st.dataframe(
                display_df.style
                    .format({"%Detected": "{:.1f}%", "%Not Detected": "{:.1f}%"})
                    .apply(lambda x: [
                        "background-color: #f0fdf4; font-weight: bold" if i == 0 else ""
                        for i in range(len(x))
                    ], axis=0),
                use_container_width=True,
                hide_index=True,
                height=420,
            )

        with col_r:
            # Pie chart
            tot_d  = summary_df["Detected"].sum()
            tot_nd = summary_df["Not Detected"].sum()
            fig_pie = go.Figure(go.Pie(
                labels=["Detected", "Not Detected"],
                values=[tot_d, tot_nd],
                marker_colors=["#ef4444", "#3b82f6"],
                hole=0.45,
                textinfo="label+percent",
                hovertemplate="%{label}: %{value:,}<extra></extra>",
            ))
            fig_pie.update_layout(
                title=f"{group_col}<br><sup>รวม {len(sub):,} ตัวอย่าง</sup>",
                margin=dict(t=50, b=10),
                height=260,
                showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # If there are detected samples → show detail
            detected_rows = sub[sub[group_col] != "Not Detect"]
            if not detected_rows.empty:
                st.markdown("#### ⚠️ ตัวอย่างที่พบสารตกค้าง")
                det_detail = (
                    detected_rows[["Sample Name", "Supplier"]]
                    .drop_duplicates()
                    .rename(columns={"Sample Name": "ชนิดตัวอย่าง"})
                    .reset_index(drop=True)
                )
                st.dataframe(det_detail, use_container_width=True, hide_index=True)
            else:
                st.success("✅ ไม่พบสารตกค้างในกลุ่มนี้")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    "<p style='text-align:center; color:#9ca3af; font-size:12px;'>"
    "Chemical Residue Testing Dashboard · QA Raw Material · Generated by Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
