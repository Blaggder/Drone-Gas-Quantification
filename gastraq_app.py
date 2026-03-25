"""
GASTRAQ – Kvantifiering av metangasutsläpp från drönarmätningar (Yt-Extrapolering)
==================================================================================
Applikation byggd i Streamlit för analys av TDLAS-mätdata från DJI M350.
Krav: pip install streamlit pandas numpy
"""

import streamlit as st
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# SIDKONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GASTRAQ – Metananalys",
    page_icon="🛸",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# STEG 1: SIDOPANEL – INSTÄLLNINGAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    
    st.title("⚙️ Inställningar")
    
    st.markdown("**1. Yt-extrapolering (Totalutsläpp)**")
    total_area = st.number_input(
        label="Deponins totala area (m²)",
        min_value=100,
        max_value=1000000,
        value=50000,
        step=1000,
        help="Används för att multiplicera det flugna medelutsläppet till ett totalutsläpp för hela anläggningen.",
    )

    st.markdown("**2. Vindmodell (GASTRAQ)**")
    v_ref = st.number_input(
        label="Vindhastighet vid marken (m/s)",
        min_value=0.1, max_value=30.0, value=3.0, step=0.1,
    )
    z_ref = st.number_input(
        label="Markmätarens höjd (m)",
        min_value=0.5, max_value=20.0, value=2.0, step=0.5,
    )
    alpha = st.number_input(
        label="Terrängfaktor (Alpha)",
        min_value=0.01, max_value=1.0, value=0.15, step=0.01,
    )
    
    st.markdown("**3. Trestegs-sköld (Brusfilter)**")
    bg_deduction = st.number_input(
        label="1. Bakgrundsavdrag (ppm*m)",
        min_value=0.0, max_value=1000.0, value=150.0, step=10.0,
        help="Golvet: Dras av från all gasdata för att få bort sensorns naturliga brus.",
    )
    min_measurements = st.number_input(
        label="2. Minsta mätningar per ruta",
        min_value=1, max_value=20, value=3, step=1,
        help="Anomalifiltret: Raderar 'laserspikar'. En ruta måste ha så här många mätpunkter för att räknas med.",
    )
    max_valid_gas = st.number_input(
        label="3. Maxvärde / Vattentak (ppm*m)",
        min_value=1000.0, max_value=100000.0, value=20000.0, step=1000.0,
        help="Taket: Värden över detta raderas helt. Filtrerar bort optiska reflektioner från vattenbryn och plåttak.",
    )


# ──────────────────────────────────────────────────────────────────────────────
# STEG 2: HUVUDYTA & FILUPPLADDNING
# ──────────────────────────────────────────────────────────────────────────────
st.title("🛸 GASTRAQ – Ytkartering & Totalutsläpp")
st.markdown(
    "Ladda upp loggfil för att automatiskt bygga rutnät, sortera bort brus (vatten/spikar) "
    "och extrapolera fram anläggningens sanna totalutsläpp baserat på drönarens medelvärde."
)

uploaded_file = st.file_uploader("📂 Välj en CSV-loggfil", type=["csv"])

# ──────────────────────────────────────────────────────────────────────────────
# MOTORN
# ──────────────────────────────────────────────────────────────────────────────
def bearbeta_data(fil, v_ref, z_ref, alpha, bg_deduction, min_measurements, max_valid_gas):
    df = pd.read_csv(fil, low_memory=False)
    antal_radpunkter = len(df)

    # 1. Koordinater & Höjd
    df["Lat"] = pd.to_numeric(df.get("Latitude RTK"), errors="coerce")
    df["Lon"] = pd.to_numeric(df.get("Longitude RTK"), errors="coerce")
    if "Latitude" in df.columns:
        df["Lat"] = df["Lat"].fillna(pd.to_numeric(df["Latitude"], errors="coerce"))
    if "Longitude" in df.columns:
        df["Lon"] = df["Lon"].fillna(pd.to_numeric(df["Longitude"], errors="coerce"))
    df["ALT:Altitude"] = pd.to_numeric(df.get("ALT:Altitude", np.nan), errors="coerce").ffill()

    # 2. Välj Gaskolumn (Prioriterar Leak/Filtered över Fast för maximal stabilitet)
    gas_prioritet = ["GAS:Leak Concentration", "GAS:Filtered Concentration", "GAS:Methane", "GAS:Fast Concentration"]
    gas_kol = next((k for k in gas_prioritet if k in df.columns), None)
    if not gas_kol:
        raise ValueError("Hittade ingen känd gaskolumn i filen.")

    # 3. Interpolera och rensa NaNs
    df["Gas_Raw"] = pd.to_numeric(df[gas_kol], errors="coerce")
    df["Gas_Interpolated"] = df["Gas_Raw"].interpolate(method="linear")
    df = df.dropna(subset=["Gas_Interpolated", "Lat", "Lon"]).copy()

    # 4. FILTER TAKET: Ta bort extrema optiska reflektioner (vatten/plåt)
    df = df[df["Gas_Interpolated"] <= max_valid_gas].copy()

    # 5. FILTER GOLVET: Bakgrundsavdrag
    df["Net_Gas_ppmm"] = df["Gas_Interpolated"] - bg_deduction
    df.loc[df["Net_Gas_ppmm"] < 0, "Net_Gas_ppmm"] = 0 # Nollställ alla negativa värden

    # 6. Bygg spatialt rutnät (AV ALL FLYGD DATA, INKLUSIVE NOLLOR FÖR SANT MEDELVÄRDE)
    df["latitude"] = df["Lat"].round(5)
    df["longitude"] = df["Lon"].round(5)
    
    grid_df = df.groupby(["latitude", "longitude"]).agg(
        Net_Gas_ppmm=("Net_Gas_ppmm", "mean"),
        ALT_Altitude=("ALT:Altitude", "mean"),
        Antal_Mätningar=("Net_Gas_ppmm", "count"),
    ).reset_index()
    grid_df = grid_df.rename(columns={"ALT_Altitude": "ALT:Altitude"})

    # 7. FILTER ANOMALIER: Rensa bort spöksignaler (rutor med för få mätningar)
    grid_df = grid_df[grid_df["Antal_Mätningar"] >= min_measurements].copy()
    
    # Spara hur stor yta vi faktiskt flög över och fick godkänd data ifrån
    flugen_yta_m2 = len(grid_df)

    # 8. Massbalans (g/h per kvadratmeter) via GASTRAQ
    grid_df["Wind_m_s"] = v_ref * ((grid_df["ALT:Altitude"] / z_ref) ** alpha)
    grid_df["Mass_g_m2"] = grid_df["Net_Gas_ppmm"] * 0.000667
    grid_df["Flux_g_h"] = grid_df["Mass_g_m2"] * grid_df["Wind_m_s"] * 3600

    # 9. Räkna ut det sanna medelvärdet för hela den flugna ytan
    medelutslapp_per_m2 = grid_df["Flux_g_h"].mean() if not grid_df.empty else 0

    # 10. SKAPA HOTSPOT-KARTAN (Filtrera bort nollorna enbart för tabell och karta)
    hotspot_df = grid_df[grid_df["Net_Gas_ppmm"] > 0].sort_values("Flux_g_h", ascending=False).reset_index(drop=True)

    return antal_radpunkter, hotspot_df, gas_kol, flugen_yta_m2, medelutslapp_per_m2


# ──────────────────────────────────────────────────────────────────────────────
# KÖRNING & RESULTAT
# ──────────────────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    with st.spinner("⏳ Bearbetar data, bygger rutnät och extrapolerar ytor..."):
        try:
            antal_radpunkter, hotspot_df, gas_kol, flugen_yta_m2, medelutslapp_per_m2 = bearbeta_data(
                uploaded_file, v_ref, z_ref, alpha, bg_deduction, min_measurements, max_valid_gas
            )
        except Exception as e:
            st.error(f"❌ Fel vid databearbetning: {e}")
            st.stop()

    # --- KPI: TOTALUTSLÄPP ---
    estimerad_total_kgh = (medelutslapp_per_m2 * total_area) / 1000
    
    st.success(f"✅ Beräkning klar! Använde gaskolumn: **{gas_kol}**")
    
    st.info(f"""
    ### 🌍 Estimerat Totalutsläpp: {estimerad_total_kgh:,.1f} kg/h
    **Uträkning:** Drönaren flög över **{flugen_yta_m2:,} m²** och mätte ett genomsnittligt utsläpp på **{medelutslapp_per_m2:.2f} g/h per m²** över den ytan (inklusive rena ytor). 
    Detta medelvärde har sedan multiplicerats med anläggningens totala yta (**{total_area:,} m²**) för att ge en uppskattning för hela deponin.
    """)

    st.divider()

    # --- KPI: HOTSPOTS ---
    col1, col2, col3 = st.columns(3)
    col1.metric("🗺️ Läckande 1x1m rutor", f"{len(hotspot_df):,}")
    värsta = hotspot_df["Flux_g_h"].iloc[0] if not hotspot_df.empty else 0
    col2.metric("🔥 Värsta Rutan (g/h)", f"{värsta:,.1f}")
    col3.metric("📊 Processade råpunkter", f"{antal_radpunkter:,}")

    # --- KARTA ---
    st.subheader("📍 Karta över detekterade Hotspots")
    if not hotspot_df.empty:
        # Gör punktstorleken dynamisk så stora läckor syns tydligare
        hotspot_df["Punktstorlek"] = (hotspot_df["Flux_g_h"] / 100) + 5
        st.map(hotspot_df, latitude="latitude", longitude="longitude", size="Punktstorlek", color="#ff0000", zoom=17)
    else:
        st.success("Inga hotspots hittades! Ytan är helt ren från läckage utifrån valda brusfilter.")

    # --- TOPPLISTA ---
    st.subheader("📋 Topplista – Värsta rutorna")
    if not hotspot_df.empty:
        visnings_df = hotspot_df[["latitude", "longitude", "Flux_g_h", "Net_Gas_ppmm", "ALT:Altitude", "Wind_m_s", "Antal_Mätningar"]].copy()
        visnings_df = visnings_df.round({
            "latitude": 5, "longitude": 5, "Flux_g_h": 1, "Net_Gas_ppmm": 1, "ALT:Altitude": 1, "Wind_m_s": 2
        })
        visnings_df.columns = ["Latitud", "Longitud", "Flöde (g/h)", "Nettogas (ppm·m)", "Höjd (m)", "Vind (m/s)", "Antal mätningar"]
        
        st.dataframe(visnings_df, use_container_width=True, height=400)
        
        csv_export = hotspot_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Ladda ner hotspots som CSV", data=csv_export, file_name="gastraq_hotspots.csv", mime="text/csv")
else:
    st.info("👆 Ladda upp din UgCS Skyhub CSV-fil för att starta analysen.")
