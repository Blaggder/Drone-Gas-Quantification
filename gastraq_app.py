"""
GASTRAQ – Kvantifiering av metangasutsläpp från drönarmätningar
================================================================
Applikation byggd i Streamlit för analys av TDLAS-mätdata från
DJI M350 med UgCS Skyhub-loggning.

Krav:
    pip install streamlit pandas numpy

Kör appen:
    streamlit run app.py
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
# STEG 1: SIDOPANEL – GASTRAQ VINDMODELLSINSTÄLLNINGAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
   
    st.title("⚙️ GASTRAQ-inställningar")
    st.markdown("**Vindmodell (log-lag profil)**")

    # Vindhastighet vid referenshöjden
    v_ref = st.number_input(
        label="Vindhastighet vid marken (m/s)",
        min_value=0.1,
        max_value=30.0,
        value=3.0,
        step=0.1,
        help="Uppmätt vindhastighet på referenshöjden z_ref.",
    )

    # Referenshöjd för vindmätaren
    z_ref = st.number_input(
        label="Markmätarens höjd (m)",
        min_value=0.5,
        max_value=20.0,
        value=2.0,
        step=0.5,
        help="Höjden ovan mark där v_ref är uppmätt.",
    )

    # Terrängfaktor (Hellman-exponent / alpha)
    alpha = st.number_input(
        label="Terrängfaktor (Alpha)",
        min_value=0.01,
        max_value=1.0,
        value=0.15,
        step=0.01,
        help="Hellman-exponenten: 0.10 = hav, 0.15 = öppen mark, 0.25 = förort.",
    )

    st.divider()
    st.caption(
        "Modell: v(z) = v_ref × (z / z_ref)^α\n\n"
        "Källa: GASTRAQ massbalansmetodik (Rees et al.)"
    )


# ──────────────────────────────────────────────────────────────────────────────
# STEG 2: HUVUDYTA & FILUPPLADDNING
# ──────────────────────────────────────────────────────────────────────────────
st.title("🛸 GASTRAQ – Metangaskvantifiering")
st.markdown(
    "Ladda upp en **UgCS Skyhub CSV-logg** från DJI M350 med TDLAS-metansensor. "
    "Appen beräknar utsläppsflöden per 1×1 m-ruta med GASTRAQ-massbalansmodellen."
)

uploaded_file = st.file_uploader(
    label="📂 Välj en CSV-loggfil",
    type=["csv"],
    help="Exporterad loggfil från UgCS Skyhub (DJI M350 + TDLAS).",
)


# ──────────────────────────────────────────────────────────────────────────────
# HJÄLPFUNKTION: Databearbetningsmotor
# ──────────────────────────────────────────────────────────────────────────────
def bearbeta_data(fil, v_ref: float, z_ref: float, alpha: float):
    """
    Läser, rensar och bearbetar en UgCS Skyhub-loggfil.

    Returnerar:
        df_raw   – Rå DataFrame direkt efter inläsning (för statistik)
        grid_df  – Aggregerat rutnät redo för visualisering
        gas_kol  – Namnet på den gaskolumn som valdes
    """

    # ── 3.1 Läs in filen ──────────────────────────────────────────────────
    df = pd.read_csv(fil, low_memory=False)

    # Spara antal råpunkter för later statistik
    antal_radpunkter = len(df)

    # ── 3.2 Koordinathantering ────────────────────────────────────────────
    # Prioritera RTK-GPS; fall tillbaka på vanlig GPS vid NaN
    df["Lat"] = pd.to_numeric(df.get("Latitude RTK"), errors="coerce")
    df["Lon"] = pd.to_numeric(df.get("Longitude RTK"), errors="coerce")

    # Fyll RTK-luckor med vanlig GPS
    if "Latitude" in df.columns:
        df["Lat"] = df["Lat"].fillna(pd.to_numeric(df["Latitude"], errors="coerce"))
    if "Longitude" in df.columns:
        df["Lon"] = df["Lon"].fillna(pd.to_numeric(df["Longitude"], errors="coerce"))

    # Forward-filla höjd (radarmätaren loggar glesare än gassen)
    if "ALT:Altitude" in df.columns:
        df["ALT:Altitude"] = pd.to_numeric(df["ALT:Altitude"], errors="coerce").ffill()
    else:
        raise ValueError("Kolumnen 'ALT:Altitude' saknas i loggfilen.")

    # ── 3.3 Smart kolumnväljare för gaskoncentration ───────────────────────
    # Sensorn har bytt exportformat; testa i prioriteringsordning
    gas_prioritet = ["GAS:Fast Concentration", "GAS:Leak Concentration", "GAS:Methane"]
    gas_kol = None
    for kandidat in gas_prioritet:
        if kandidat in df.columns:
            gas_kol = kandidat
            break

    if gas_kol is None:
        raise ValueError(
            f"Ingen gaskolumn hittades. Sökte efter: {gas_prioritet}. "
            "Kontrollera att rätt loggfil laddats upp."
        )

    df["Gas_Raw"] = pd.to_numeric(df[gas_kol], errors="coerce")

    # ── 3.4 Tidsinterpolering (40 Hz → inga luckor) ───────────────────────
    # Linjär interpolering binder ihop mätpunkter från sensorer som
    # loggar på olika tidsstämplar (typiskt problemet med UgCS Skyhub)
    df["Gas_Interpolated"] = df["Gas_Raw"].interpolate(method="linear")

    # Ta bort rader som ändå saknar gas, lat eller lon
    df = df.dropna(subset=["Gas_Interpolated", "Lat", "Lon"])

    # ── 3.5 Dynamiskt brusfilter (höjdbaserat) ────────────────────────────
    # Bakgrundsnivån ökar med höjden (sensorfel & atmosfärisk bakgrund).
    # Allt under gränsen klassas som bakgrundsbrus.
    df["Background_Limit"] = df["ALT:Altitude"] * 18.0
    df_hotspots = df[df["Gas_Interpolated"] > df["Background_Limit"]].copy()

    if df_hotspots.empty:
        raise ValueError(
            "Inga hotspots detekterades efter brusfiltrering. "
            "Pröva att sänka bakgrundsfaktorn (18.0) eller kontrollera filen."
        )

    # ── 3.6 Nettogas (ppm·m, rensat från höjdberoende bakgrund) ──────────
    df_hotspots["Net_Gas_ppmm"] = (
        df_hotspots["Gas_Interpolated"] - (df_hotspots["ALT:Altitude"] * 2.0)
    )
    # Negativa värden sätts till 0 (kan uppstå vid extremt låga koncentrationer)
    df_hotspots["Net_Gas_ppmm"] = df_hotspots["Net_Gas_ppmm"].clip(lower=0)

    # ── 3.7 Spatialt rutnät (1×1 m approximation via 5 decimaler) ────────
    # 0.00001° lat ≈ 1.11 m; 0.00001° lon ≈ 0.64–1.11 m beroende på latitud.
    # Tillräckligt för att skapa meningsfulla 1×1 m-rutor i fältmiljö.
    df_hotspots["latitude"] = df_hotspots["Lat"].round(5)
    df_hotspots["longitude"] = df_hotspots["Lon"].round(5)

    # Aggregera per ruta
    grid_df = (
        df_hotspots.groupby(["latitude", "longitude"])
        .agg(
            Net_Gas_ppmm=("Net_Gas_ppmm", "mean"),
            ALT_Altitude=("ALT:Altitude", "mean"),
            Antal_Mätningar=("Net_Gas_ppmm", "count"),
        )
        .reset_index()
    )

    # Byt tillbaka kolumnnamnet för tydlighet
    grid_df = grid_df.rename(columns={"ALT_Altitude": "ALT:Altitude"})

    # ── 3.8 Massbalans – GASTRAQ-modellen ────────────────────────────────
    # Beräkna vindhastighet på drönarhöjden med log-lag-profil
    grid_df["Wind_m_s"] = v_ref * ((grid_df["ALT:Altitude"] / z_ref) ** alpha)

    # Omvandla ppm·m till g/m² (metan: densitet ≈ 0.000667 g/ppm·m vid STP)
    grid_df["Mass_g_m2"] = grid_df["Net_Gas_ppmm"] * 0.000667

    # Massflöde per sekund (massa × vindhastighet × 1 m bredd)
    grid_df["Flux_g_s"] = grid_df["Mass_g_m2"] * grid_df["Wind_m_s"] * 1.0

    # Massflöde per timme
    grid_df["Flux_g_h"] = grid_df["Flux_g_s"] * 3600

    # ── 3.9 Sortera på värsta flöde ───────────────────────────────────────
    grid_df = grid_df.sort_values("Flux_g_h", ascending=False).reset_index(drop=True)

    return antal_radpunkter, grid_df, gas_kol


# ──────────────────────────────────────────────────────────────────────────────
# STEG 3 + 4: TRIGGA BEARBETNING & VISA RESULTAT
# ──────────────────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    with st.spinner("⏳ Bearbetar loggdata – interpolerar sensorer och beräknar flöden…"):
        try:
            antal_radpunkter, grid_df, gas_kol = bearbeta_data(
                uploaded_file, v_ref, z_ref, alpha
            )
        except ValueError as e:
            st.error(f"❌ Fel vid bearbetning: {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Oväntat fel: {e}")
            st.stop()

    st.success(f"✅ Klar! Gaskolumn som användes: **{gas_kol}**")

    # ── 4.1 Nyckeltal (metric-boxar) ──────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="🗺️ Unika Hotspots (1×1 m)",
            value=f"{len(grid_df):,}",
            help="Antal unika rutor i det spatiala rutnätet med detekterade utsläpp.",
        )
    with col2:
        värsta_flux = grid_df["Flux_g_h"].iloc[0] if not grid_df.empty else 0
        st.metric(
            label="🔥 Värsta rutan (g/h)",
            value=f"{värsta_flux:,.1f}",
            help="Högsta beräknade massflöde i en enskild 1×1 m-ruta.",
        )
    with col3:
        st.metric(
            label="📊 Råpunkter totalt",
            value=f"{antal_radpunkter:,}",
            help="Totalt antal datarader i originalfilen (inkl. NaN).",
        )

    st.divider()

    # ── 4.2 Karta ─────────────────────────────────────────────────────────
    st.subheader("🗺️ Karta över Hotspots")
    st.caption(
        "Varje punkt representerar en 1×1 m-ruta med detekterat metanutsläpp. "
        "Zoomfunktion och panorering stöds i kartan."
    )

    # st.map kräver kolumnerna 'latitude' och 'longitude'
    st.map(grid_df[["latitude", "longitude"]], zoom=17, use_container_width=True)

    st.divider()

    # ── 4.3 Topplista (DataFrame) ──────────────────────────────────────────
    st.subheader("📋 Topplista – Värsta 1×1 m-rutorna")
    st.caption("Sorterat efter beräknat massflöde (g/h), fallande.")

    # Välj och formatera de kolumner som är relevanta för fältoperatören
    visnings_df = grid_df[
        [
            "latitude",
            "longitude",
            "Flux_g_h",
            "Net_Gas_ppmm",
            "ALT:Altitude",
            "Wind_m_s",
            "Antal_Mätningar",
        ]
    ].copy()

    # Avrundning för läsbarhet
    visnings_df = visnings_df.round(
        {
            "latitude": 5,
            "longitude": 5,
            "Flux_g_h": 1,
            "Net_Gas_ppmm": 2,
            "ALT:Altitude": 1,
            "Wind_m_s": 2,
        }
    )

    # Byt till svenska kolumnrubriker
    visnings_df.columns = [
        "Latitud",
        "Longitud",
        "Flöde (g/h)",
        "Nettogas (ppm·m)",
        "Höjd (m)",
        "Vind (m/s)",
        "Antal mätningar",
    ]

    st.dataframe(
        visnings_df,
        use_container_width=True,
        height=420,
    )

    # Möjlighet att ladda ner resultatet som CSV
    csv_export = grid_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Ladda ner rutnät som CSV",
        data=csv_export,
        file_name="gastraq_resultat.csv",
        mime="text/csv",
    )

else:
    # Välkomstmeddelande när ingen fil laddats upp
    st.info(
        "👆 Ladda upp en UgCS Skyhub-loggfil i CSV-format för att starta analysen.\n\n"
        "**Förväntade kolumner:**\n"
        "- `Latitude RTK` / `Longitude RTK` (eller `Latitude` / `Longitude`)\n"
        "- `ALT:Altitude`\n"
        "- `GAS:Fast Concentration` (eller `GAS:Leak Concentration` / `GAS:Methane`)",
        icon="ℹ️",
    )

    # Visa ett pedagogiskt flödesschema i sidopanelen
    with st.expander("ℹ️ Om GASTRAQ-modellen"):
        st.markdown(
            """
            ### Massbalansmetodik

            GASTRAQ-modellen beräknar metanflöde per ytenhet med formeln:

            ```
            v(z) = v_ref × (z / z_ref)^α          [Hellmans vindprofil]
            Mass  = Net_Gas_ppmm × 0.000667        [g/m²]
            Flux  = Mass × v(z) × 1.0              [g/s per m-bredd]
            ```

            Parametrarna `v_ref`, `z_ref` och `α` ställs in i sidopanelen.

            ### Brusfilter

            Bakgrundsgränsen är höjdberoende:
            ```
            Background_Limit = ALT:Altitude × 18.0
            ```
            Endast mätningar som överstiger denna gräns klassas som hotspots.

            ### Datakällor
            - DJI M350 RTK med TDLAS-metansensor
            - Loggning via UgCS Skyhub (upp till 40 Hz)
            - RTK-GPS med cm-noggrannhet
            """
        )
