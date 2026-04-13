import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import statsmodels.api as sm


# setări pagină
st.set_page_config(
    page_title="Analiza turismului internațional",
    layout="wide"
)

# titlu aplicație
st.title("Analiza fluxurilor turistice și oportunități de dezvoltare turistică")

# descriere scurtă
st.markdown(
    """
    Aplicația analizează evoluția turismului internațional pe baza unor indicatori World Bank:
    - sosiri turistice internaționale
    - venituri din turism
    - populație totală
    - PIB
    """
)

# stil minim
st.markdown(
    """
    <style>
    .custom-box {
        padding: 12px;
        border-radius: 10px;
        background-color: #f4f6f8;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# fișiere folosite
GDP_FILE = "API_NY.GDP.MKTP.CD_DS2_en_csv_v2_24.csv"
POP_FILE = "API_SP.POP.TOTL_DS2_en_csv_v2_58.csv"
ARR_FILE = "API_ST.INT.ARVL_DS2_en_csv_v2_585.csv"
REC_FILE = "API_ST.INT.RCPT.CD_DS2_en_csv_v2_220.csv"


# verifică dacă există toate fișierele
def check_files_exist():
    required_files = [GDP_FILE, POP_FILE, ARR_FILE, REC_FILE]
    missing = [f for f in required_files if not Path(f).exists()]
    return missing


# citește csv-ul și păstrează doar anii 2010-2020
def load_world_bank_csv(file_path: str, value_name: str, start_year: int = 2010, end_year: int = 2020) -> pd.DataFrame:
    df = pd.read_csv(file_path, skiprows=4)

    year_cols = [str(year) for year in range(start_year, end_year + 1) if str(year) in df.columns]

    df = df[["Country Name", "Country Code"] + year_cols].copy()

    df_long = df.melt(
        id_vars=["Country Name", "Country Code"],
        value_vars=year_cols,
        var_name="Year",
        value_name=value_name
    )

    df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")
    df_long[value_name] = pd.to_numeric(df_long[value_name], errors="coerce")

    return df_long


# scoate regiunile agregate
def remove_aggregate_regions(df: pd.DataFrame) -> pd.DataFrame:
    aggregate_codes = {
        "AFE", "AFW", "ARB", "CEB", "CSS", "EAP", "EAR", "EAS", "ECA", "ECS",
        "EMU", "EUU", "FCS", "HIC", "HPC", "IBD", "IBT", "IDA", "IDX", "INX",
        "LAC", "LCN", "LDC", "LIC", "LMC", "LMY", "LTE", "MEA", "MIC", "MNA",
        "NAC", "OEC", "OSS", "PRE", "PST", "SAS", "SSA", "SSF", "SST", "TEA",
        "TEC", "TLA", "TMN", "TSA", "TSS", "UMC", "WLD"
    }
    return df[~df["Country Code"].isin(aggregate_codes)].copy()


# limitează valorile extreme
def cap_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    result = df.copy()
    series = result[column].dropna()

    if len(series) == 0:
        return result

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    result[column] = result[column].clip(lower=lower, upper=upper)
    return result


# creează nivel turistic
def assign_tourism_level(value: float) -> str:
    if pd.isna(value):
        return "Necunoscut"
    if value < 100:
        return "Scazut"
    if value < 500:
        return "Mediu"
    return "Ridicat"


# pregătește setul final
@st.cache_data
def prepare_dataset() -> pd.DataFrame:
    arrivals = load_world_bank_csv(ARR_FILE, "Arrivals")
    receipts = load_world_bank_csv(REC_FILE, "Receipts_USD")
    population = load_world_bank_csv(POP_FILE, "Population")
    gdp = load_world_bank_csv(GDP_FILE, "GDP_USD")

    df = arrivals.merge(receipts, on=["Country Name", "Country Code", "Year"], how="outer")
    df = df.merge(population, on=["Country Name", "Country Code", "Year"], how="outer")
    df = df.merge(gdp, on=["Country Name", "Country Code", "Year"], how="outer")

    df = remove_aggregate_regions(df)

    # completează lipsurile
    numeric_cols = ["Arrivals", "Receipts_USD", "Population", "GDP_USD"]
    for col in numeric_cols:
        df[col] = df.groupby("Country Code")[col].transform(lambda s: s.interpolate(limit_direction="both"))
        df[col] = df[col].fillna(df[col].median())

    # calculează indicatori noi
    df["Receipts_per_Tourist"] = np.where(df["Arrivals"] > 0, df["Receipts_USD"] / df["Arrivals"], np.nan)
    df["Tourists_per_1000"] = np.where(df["Population"] > 0, (df["Arrivals"] / df["Population"]) * 1000, np.nan)
    df["Tourism_GDP_pct"] = np.where(df["GDP_USD"] > 0, (df["Receipts_USD"] / df["GDP_USD"]) * 100, np.nan)

    # repară valori invalide
    for col in ["Receipts_per_Tourist", "Tourists_per_1000", "Tourism_GDP_pct"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(df[col].median())

    # taie extremele
    for col in ["Arrivals", "Receipts_USD", "Receipts_per_Tourist", "Tourists_per_1000", "Tourism_GDP_pct"]:
        df = cap_outliers_iqr(df, col)

    # transformă într-o categorie
    df["Tourism_Level"] = df["Tourists_per_1000"].apply(assign_tourism_level)

    # codificare numerică
    le = LabelEncoder()
    df["Tourism_Level_Encoded"] = le.fit_transform(df["Tourism_Level"])

    # codificare dummy
    dummies = pd.get_dummies(df["Tourism_Level"], prefix="Tourism", drop_first=True)
    df = pd.concat([df, dummies], axis=1)

    return df


# oprește dacă lipsesc fișiere
missing_files = check_files_exist()
if missing_files:
    st.error("Lipsesc următoarele fișiere CSV din folderul proiectului:")
    for file_name in missing_files:
        st.write(f"- {file_name}")
    st.stop()


# încarcă datele
df = prepare_dataset()


# filtre din stânga
section = st.sidebar.radio(
    "Navigați la:",
    [
        "Introducere",
        "Date curate",
        "Statistici pe an",
        "Top tari",
        "Grafice",
        "Clusterizare",
        "Regresie",
        "Concluzii"
    ]
)

years = sorted(df["Year"].dropna().unique().tolist())
selected_year = st.sidebar.selectbox("Selectează anul", years, index=len(years) - 1)

df_year = df[df["Year"] == selected_year].copy()

top_n = st.sidebar.slider("Număr țări afișate", 5, 25, 10)

selected_metric = st.sidebar.selectbox(
    "Indicator pentru top",
    [
        "Arrivals",
        "Receipts_USD",
        "Receipts_per_Tourist",
        "Tourists_per_1000",
        "Tourism_GDP_pct"
    ]
)

st.sidebar.write("An selectat:", selected_year)
st.sidebar.write("Număr observații pentru anul selectat:", len(df_year))


# text introductiv
if section == "Introducere":
    st.header("Ce analizează proiectul")

    st.markdown(
        """
        Proiectul urmărește performanța turistică a mai multor țări în perioada 2010-2020.

        Sunt analizate:
        - volumul sosirilor turistice
        - veniturile generate de turism
        - raportarea la populație
        - raportarea la PIB
        """
    )

    st.subheader("Surse de date")
    st.write("- World Bank: sosiri turistice")
    st.write("- World Bank: venituri din turism")
    st.write("- World Bank: populație")
    st.write("- World Bank: PIB")

    st.subheader("Ce tehnici sunt folosite")
    st.markdown(
        """
        - afișare în Streamlit
        - curățare de date
        - tratarea valorilor lipsă
        - limitarea valorilor extreme
        - codificare de variabile
        - scalare de variabile
        - agregare cu pandas
        - clusterizare KMeans
        - regresie multiplă cu statsmodels
        """
    )


# arată datele finale și curățarea
elif section == "Date curate":
    st.header("Cum arată datele după prelucrare")

    st.subheader("Primele rânduri")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Dimensiunea setului")
    c1, c2, c3 = st.columns(3)
    c1.metric("Număr observații", len(df))
    c2.metric("Număr țări", df["Country Code"].nunique())
    c3.metric("Interval ani", f"{int(df['Year'].min())} - {int(df['Year'].max())}")

    st.subheader("Lipsuri rămase")
    st.dataframe(df.isnull().sum().to_frame("Valori lipsă"), use_container_width=True)

    st.subheader("Date doar pentru anul selectat")
    st.dataframe(df_year.head(20), use_container_width=True)

    st.markdown(
        """
        Aici se vede setul final după:
        - combinarea fișierelor
        - completarea valorilor lipsă
        - calculul indicatorilor noi
        - limitarea valorilor extreme
        """
    )


# statistici pentru anul ales
elif section == "Statistici pe an":
    st.header(f"Indicatori statistici pentru anul {selected_year}")

    desc = df_year[
        [
            "Arrivals",
            "Receipts_USD",
            "Population",
            "GDP_USD",
            "Receipts_per_Tourist",
            "Tourists_per_1000",
            "Tourism_GDP_pct"
        ]
    ].describe().T

    st.subheader("Statistici descriptive")
    st.dataframe(desc, use_container_width=True)

    st.subheader("Valori medii pe întreg intervalul")
    yearly_stats = (
        df.groupby("Year")
        .agg(
            Total_Arrivals=("Arrivals", "sum"),
            Total_Receipts=("Receipts_USD", "sum"),
            Avg_Receipts_per_Tourist=("Receipts_per_Tourist", "mean"),
            Avg_Tourists_per_1000=("Tourists_per_1000", "mean"),
            Avg_Tourism_GDP_pct=("Tourism_GDP_pct", "mean")
        )
        .reset_index()
    )
    st.dataframe(yearly_stats, use_container_width=True)

    st.markdown(
        """
        Prima tabelă se schimbă când alegi alt an.
        A doua tabelă rămâne pe toți anii pentru a arăta evoluția completă.
        """
    )


# topuri pentru anul ales
elif section == "Top tari":
    st.header(f"Clasamente pentru anul {selected_year}")

    st.write(f"Se afișează primele {top_n} țări după indicatorul {selected_metric}.")

    ranking = (
        df_year[
            [
                "Country Name",
                "Country Code",
                "Arrivals",
                "Receipts_USD",
                "Receipts_per_Tourist",
                "Tourists_per_1000",
                "Tourism_GDP_pct",
                "Tourism_Level"
            ]
        ]
        .sort_values(selected_metric, ascending=False)
        .head(top_n)
    )

    st.subheader(f"Top {top_n} țări")
    st.dataframe(ranking, use_container_width=True)

    st.subheader("Țări cu posibil potențial de dezvoltare")
    potential_df = df_year[
        (df_year["Tourists_per_1000"] > df_year["Tourists_per_1000"].median()) &
        (df_year["Receipts_per_Tourist"] < df_year["Receipts_per_Tourist"].median())
    ][
        [
            "Country Name",
            "Arrivals",
            "Receipts_USD",
            "Receipts_per_Tourist",
            "Tourists_per_1000",
            "Tourism_GDP_pct"
        ]
    ].sort_values("Tourists_per_1000", ascending=False)

    st.dataframe(potential_df.head(top_n), use_container_width=True)

    st.markdown(
        """
        Schimbarea anului modifică țările comparate.
        Schimbarea numărului de țări modifică lungimea clasamentului.
        """
    )


# grafice influențate de filtre
elif section == "Grafice":
    st.header(f"Grafice pentru anul {selected_year}")

    yearly_stats = (
        df.groupby("Year")
        .agg(
            Total_Arrivals=("Arrivals", "sum"),
            Total_Receipts=("Receipts_USD", "sum")
        )
        .reset_index()
    )

    st.subheader("Evoluția sosirilor totale")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(yearly_stats["Year"], yearly_stats["Total_Arrivals"], marker="o")
    ax1.set_xlabel("An")
    ax1.set_ylabel("Sosiri turistice")
    ax1.set_title("Sosiri turistice totale pe ani")
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

    st.subheader("Evoluția veniturilor totale")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(yearly_stats["Year"], yearly_stats["Total_Receipts"], marker="o")
    ax2.set_xlabel("An")
    ax2.set_ylabel("Venituri din turism (USD)")
    ax2.set_title("Venituri din turism pe ani")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    ranking = (
        df_year[["Country Name", selected_metric]]
        .sort_values(selected_metric, ascending=False)
        .head(top_n)
        .sort_values(selected_metric, ascending=True)
    )

    st.subheader(f"Top {top_n} țări în anul {selected_year}")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.barh(ranking["Country Name"], ranking[selected_metric])
    ax3.set_xlabel(selected_metric)
    ax3.set_ylabel("Țara")
    ax3.set_title(f"Top țări după {selected_metric}")
    st.pyplot(fig3)

    st.subheader(f"Relația dintre sosiri și venituri în {selected_year}")
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ax4.scatter(df_year["Arrivals"], df_year["Receipts_USD"])
    ax4.set_xlabel("Sosiri turistice")
    ax4.set_ylabel("Venituri din turism")
    ax4.set_title(f"Sosiri vs venituri - {selected_year}")
    ax4.grid(True, alpha=0.3)
    st.pyplot(fig4)

    st.markdown(
        """
        Primele două grafice arată evoluția pe toată perioada.
        Ultimele două grafice se schimbă direct după anul și topul ales.
        """
    )


# grupează țările pentru anul ales
elif section == "Clusterizare":
    st.header(f"Clusterizare pentru anul {selected_year}")

    cluster_df = df_year[
        [
            "Country Name",
            "Arrivals",
            "Receipts_USD",
            "Receipts_per_Tourist",
            "Tourists_per_1000",
            "Tourism_GDP_pct"
        ]
    ].dropna().copy()

    feature_cols = [
        "Arrivals",
        "Receipts_USD",
        "Receipts_per_Tourist",
        "Tourists_per_1000",
        "Tourism_GDP_pct"
    ]

    st.subheader("Date scalate")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_df[feature_cols])

    scaled_preview = pd.DataFrame(X_scaled, columns=feature_cols).head(10)
    st.dataframe(scaled_preview, use_container_width=True)

    st.subheader("Rezultatul KMeans")
    k = st.slider("Număr clustere", min_value=2, max_value=6, value=3)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_df["Cluster"] = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, cluster_df["Cluster"])
    st.metric("Silhouette Score", f"{score:.3f}")

    st.dataframe(cluster_df.head(20), use_container_width=True)

    st.subheader("Graficul clusterelor")
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    ax5.scatter(
        cluster_df["Tourists_per_1000"],
        cluster_df["Receipts_per_Tourist"],
        c=cluster_df["Cluster"]
    )
    ax5.set_xlabel("Turiști la 1000 locuitori")
    ax5.set_ylabel("Venit per turist")
    ax5.set_title("Țări grupate după profil turistic")
    ax5.grid(True, alpha=0.3)
    st.pyplot(fig5)

    st.subheader("Media pe fiecare cluster")
    cluster_summary = cluster_df.groupby("Cluster")[feature_cols].mean().round(2).reset_index()
    st.dataframe(cluster_summary, use_container_width=True)

    st.markdown(
        """
        Când schimbi anul, se schimbă și țările care intră în clusterizare,
        deci se poate modifica și structura clusterelor.
        """
    )


# estimează legătura dintre indicatori
elif section == "Regresie":
    st.header(f"Regresie multiplă pentru anul {selected_year}")

    reg_df = df_year[
        [
            "Receipts_USD",
            "Arrivals",
            "Population",
            "GDP_USD",
            "Tourism_Level_Encoded"
        ]
    ].dropna().copy()

    if len(reg_df) < 10:
        st.warning("Nu există suficiente observații pentru regresie în anul selectat.")
    else:
        reg_df["log_Receipts_USD"] = np.log1p(reg_df["Receipts_USD"])
        reg_df["log_Arrivals"] = np.log1p(reg_df["Arrivals"])
        reg_df["log_Population"] = np.log1p(reg_df["Population"])
        reg_df["log_GDP_USD"] = np.log1p(reg_df["GDP_USD"])

        X = reg_df[["log_Arrivals", "log_Population", "log_GDP_USD", "Tourism_Level_Encoded"]]
        y = reg_df["log_Receipts_USD"]

        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()

        st.subheader("Rezultatul modelului")
        st.text(model.summary().as_text())

        st.markdown(
            """
            Modelul arată cum sunt asociate veniturile din turism cu:
            - numărul de turiști
            - populația
            - PIB-ul
            - intensitatea turistică
            """
        )


# încheie analiza
elif section == "Concluzii":
    st.header("Ce arată analiza")

    st.markdown(
        """
        Rezultatele indică diferențe importante între țări în ceea ce privește:
        - volumul turismului
        - venitul obținut per turist
        - ponderea turismului în economie
        - profilul general al pieței turistice

        Schimbarea anului permite comparații între momente diferite din perioada 2010-2020.
        Schimbarea numărului de țări ajută la extinderea sau restrângerea clasamentelor analizate.
        """
    )

    st.success(
        """
        Cerințe Python acoperite:
        - Streamlit
        - valori lipsă
        - valori extreme
        - codificare
        - scalare
        - statistici și agregări
        - groupby
        - scikit-learn
        - statsmodels
        """
    )