import os
import pickle
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import joblib
except Exception:
    joblib = None


# =========================================================
# STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Hotel Revenue Forecasting Dashboard",
    layout="wide"
)


# =========================================================
# HOTEL CONFIGURATION
# Inventory tidak menjadi input user.
# Ubah hanya pada code apabila inventory hotel berubah.
# =========================================================

HOTEL_TOTAL_ROOMS = 89


# =========================================================
# BASIC HELPERS
# =========================================================

def safe_div(a, b):
    if b is None or pd.isna(b) or b == 0:
        return 0

    return a / b


def safe_div_series(a, b):
    return (
        a / b.replace(0, np.nan)
    ).replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)


def format_currency(value):
    try:
        if pd.isna(value):
            return "Rp 0"

        return f"Rp {float(value):,.0f}"

    except Exception:
        return "Rp 0"


def format_number(value):
    try:
        if pd.isna(value):
            return "0"

        return f"{float(value):,.0f}"

    except Exception:
        return "0"


def format_percent(value):
    try:
        if pd.isna(value):
            return "0.00%"

        return f"{float(value):,.2f}%"

    except Exception:
        return "0.00%"


def format_factor(value):
    try:
        if pd.isna(value):
            return "0.00x"

        return f"{float(value):,.2f}x"

    except Exception:
        return "0.00x"


def format_plain_integer(value):
    try:
        if pd.isna(value):
            return ""

        return str(int(float(value)))

    except Exception:
        return str(value)


def format_state_value(value):
    try:
        if pd.isna(value):
            return ""

        float_value = float(value)

        if float_value.is_integer():
            return str(int(float_value))

        return str(float_value)

    except Exception:
        return str(value)


def format_dataframe_for_display(df):
    output = df.copy()

    state_cols = [
        "Revenue State",
        "Most Likely Revenue State",
        "Dominant Revenue State",
        "Most Likely State"
    ]

    plain_integer_cols = [
        "Year",
        "Month",
        "Day",
        "Quarter",
        "Day Of Week Number",
        "Days In Month",
        "Elapsed Days",
        "Remaining Days",
        "Known Booking Rows",
        "Actual Running Rows",
        "State_Days"
    ]

    currency_exact_cols = [
        "Current Room Revenue",
        "Current Room ADR",
        "Actual Running Revenue",
        "Actual Running Average Room Rate",
        "OTB Revenue as of Cutoff",
        "Known Revenue",
        "Known Future Revenue",
        "Known Average Room Rate",
        "Budget Revenue",
        "Revenue Remaining to Budget",
        "Projected Revenue from Naive Bayes",
        "Projected Revenue from Linear Regression",
        "Projected Revenue",
        "Adjusted Projected Revenue",
        "Base Projected Revenue",
        "Adjustment Revenue",
        "Revenue Gap",
        "Adjusted Revenue Gap",
        "Raw Projected Revenue from Naive Bayes",
        "Raw Projected Revenue from Linear Regression",
        "Raw Projected Revenue from Naive Bayes before Cap",
        "Model Projected Revenue from Linear Regression before OTB Floor",
        "Model Projected Revenue from Naive Bayes before OTB Floor",
        "Raw Average Room Rate before Cap",
        "Average Room Rate Cap",
        "Projected Average Room Rate",
        "Adjusted Average Room Rate",
        "Budget Average Room Rate",
        "Benchmark Average Room Rate",
        "Minimum Revenue in State",
        "Maximum Revenue in State",
        "Average Revenue in State",
        "Expected Revenue from Naive Bayes",
        "Forecast Revenue from Naive Bayes",
        "Forecast Revenue from Linear Regression",
        "Revenue MA7",
        "Revenue Moving Average 7 Days",
        "Projected Revenue Per Available Room",
        "Budget Revenue Per Available Room",
        "Adjusted Revenue Per Available Room"
    ]

    percent_cols = [
        "Revenue Achievement %",
        "Adjusted Revenue Achievement %",
        "Rooms Sold Achievement %",
        "Adjusted Rooms Sold Achievement %",
        "Average Room Rate Achievement %",
        "Adjusted Average Room Rate Achievement %",
        "Projected Occupancy %",
        "Adjusted Occupancy %",
        "Budget Occupancy %",
        "OTB Revenue Achievement",
        "Naive Bayes Average Confidence %",
        "Revenue State Confidence %",
        "Dominant Revenue State Confidence %"
    ]

    factor_cols = [
        "Demand Factor",
        "Historical Same-Month Factor",
        "Gentle OTB Strength Factor",
        "OTB Strength Raw"
    ]

    number_keywords = [
        "Rooms Sold",
        "Room Capacity",
        "Total Rooms",
        "Inventory",
        "Additional",
        "Adjustment Rooms Sold",
        "Room Nights",
        "Known Booking Rows",
        "Actual Running Rows",
        "Rows",
        "RN",
        "Capacity"
    ]

    for col in output.columns:
        if col in state_cols:
            output[col] = output[col].apply(
                format_state_value
            )
            continue

        if col in plain_integer_cols:
            output[col] = output[col].apply(
                format_plain_integer
            )
            continue

        if pd.api.types.is_bool_dtype(output[col]):
            output[col] = output[col].map({
                True: "Yes",
                False: "No"
            })
            continue

        if not pd.api.types.is_numeric_dtype(output[col]):
            continue

        col_name = str(col)

        if col in currency_exact_cols:
            output[col] = output[col].apply(
                format_currency
            )

        elif col in percent_cols or "%" in col_name:
            output[col] = output[col].apply(
                format_percent
            )

        elif col in factor_cols:
            output[col] = output[col].apply(
                format_factor
            )

        elif any(
            keyword in col_name
            for keyword in number_keywords
        ):
            output[col] = output[col].apply(
                format_number
            )

        else:
            output[col] = output[col].apply(
                lambda value: (
                    f"{float(value):,.2f}"
                    if pd.notna(value)
                    else ""
                )
            )

    return output


# =========================================================
# LOAD SAVED MODEL
# =========================================================

def load_pickle_file(path):
    try:
        with open(path, "rb") as file:
            return pickle.load(file)

    except Exception:
        if joblib is not None:
            return joblib.load(path)

        raise


def normalize_state_reference(bin_reference):
    state_reference = bin_reference.copy()

    rename_map = {}

    if "mean" in state_reference.columns:
        rename_map["mean"] = (
            "Average Revenue in State"
        )

    if "min" in state_reference.columns:
        rename_map["min"] = (
            "Minimum Revenue in State"
        )

    if "max" in state_reference.columns:
        rename_map["max"] = (
            "Maximum Revenue in State"
        )

    if rename_map:
        state_reference = state_reference.rename(
            columns=rename_map
        )

    if "Revenue State" not in state_reference.columns:
        alternative_state_cols = [
            "revenue_state",
            "Revenue_State",
            "state",
            "State",
            "RevenueState"
        ]

        for col in alternative_state_cols:
            if col in state_reference.columns:
                state_reference = (
                    state_reference.rename(
                        columns={
                            col: "Revenue State"
                        }
                    )
                )
                break

    if "Revenue State" not in state_reference.columns:
        state_reference = (
            state_reference
            .reset_index()
            .rename(
                columns={
                    "index": "Revenue State"
                }
            )
        )

    if (
        "Average Revenue in State"
        not in state_reference.columns
    ):
        raise ValueError(
            "bin_reference.pkl harus memiliki "
            "kolom mean atau Average Revenue in State."
        )

    if (
        "Minimum Revenue in State"
        not in state_reference.columns
    ):
        state_reference[
            "Minimum Revenue in State"
        ] = 0

    if (
        "Maximum Revenue in State"
        not in state_reference.columns
    ):
        state_reference[
            "Maximum Revenue in State"
        ] = 0

    return state_reference


@st.cache_resource
def load_saved_model_bundle(model_dir="Model"):
    paths = {
        "linear_model": os.path.join(
            model_dir,
            "linear_regression_model_p2.pkl"
        ),
        "linear_scaler": os.path.join(
            model_dir,
            "linear_scaler_p2.pkl"
        ),
        "linear_features": os.path.join(
            model_dir,
            "linear_features.pkl"
        ),
        "nb_model": os.path.join(
            model_dir,
            "naive_bayes_model.pkl"
        ),
        "nb_scaler": os.path.join(
            model_dir,
            "naive_bayes_scaler.pkl"
        ),
        "nb_features": os.path.join(
            model_dir,
            "naive_bayes_features.pkl"
        ),
        "bin_reference": os.path.join(
            model_dir,
            "bin_reference.pkl"
        )
    }

    missing_paths = [
        path
        for path in paths.values()
        if not os.path.exists(path)
    ]

    if missing_paths:
        raise FileNotFoundError(
            "File model tidak ditemukan: " +
            ", ".join(missing_paths)
        )

    return (
        load_pickle_file(paths["linear_model"]),
        load_pickle_file(paths["linear_scaler"]),
        list(
            load_pickle_file(
                paths["linear_features"]
            )
        ),
        load_pickle_file(paths["nb_model"]),
        load_pickle_file(paths["nb_scaler"]),
        list(
            load_pickle_file(
                paths["nb_features"]
            )
        ),
        normalize_state_reference(
            load_pickle_file(
                paths["bin_reference"]
            )
        )
    )


# =========================================================
# BUDGET MASTER
# =========================================================

def build_budget_df(total_rooms):
    months = [
        "Jan", "Feb", "Mar", "Apr",
        "May", "Jun", "Jul", "Aug",
        "Sep", "Oct", "Nov", "Dec"
    ]

    rooms_sold_budget = {
        2023: [
            1136, 1603, 1503, 838,
            1711, 1140, 1502, 1284,
            1840, 1832, 1852, 1538
        ],
        2024: [
            1237, 1449, 1512, 1307,
            1617, 1521, 1432, 2171,
            2011, 1818, 1931, 1825
        ],
        2025: [
            1598, 1690, 1604, 1644,
            1818, 1732, 2040, 1932,
            1610, 1924, 2060, 1783
        ],
        2026: [
            1677, 1450, 1344, 1800,
            1875, 1872, 1904, 2081,
            2176, 2111, 2079, 1886
        ]
    }

    revenue_budget = {
        2023: [
            1632547042, 2134607087,
            2441014872, 1410250277,
            2534710951, 1737421989,
            2474222357, 2023501018,
            2893539854, 2734679290,
            3039867030, 2270702220
        ],
        2024: [
            1802719112, 2164260575,
            2539859918, 2106118031,
            2451727017, 2477207176,
            2251012539, 3290025117,
            3052494741, 2738642068,
            2816336956, 2761873954
        ],
        2025: [
            2528193052, 2683712962,
            2436235087, 2550792596,
            2933540889, 2696780633,
            3170702517, 3068785674,
            2560380352, 2977876794,
            3095777907, 2705115000
        ],
        2026: [
            2756770797, 2292243643,
            2155363741, 2839362146,
            3046819287, 2997522675,
            3122191612, 3383946391,
            3490878797, 3352713798,
            3237569378, 3008709608
        ]
    }

    rows = []

    for year in rooms_sold_budget:
        for index, month_name in enumerate(months):
            month = index + 1

            days = pd.Timestamp(
                year,
                month,
                1
            ).days_in_month

            capacity = total_rooms * days

            budget_rooms = (
                rooms_sold_budget[year][index]
            )

            budget_revenue = (
                revenue_budget[year][index]
            )

            rows.append({
                "Year": year,
                "Month": month,
                "Month Name": month_name,
                "Total Rooms / Hotel Inventory":
                    total_rooms,
                "Days In Month": days,
                "Monthly Room Capacity": capacity,
                "Budget Rooms Sold": budget_rooms,
                "Budget Revenue": budget_revenue,
                "Budget Average Room Rate":
                    safe_div(
                        budget_revenue,
                        budget_rooms
                    ),
                "Budget Occupancy %":
                    safe_div(
                        budget_rooms,
                        capacity
                    ) * 100,
                "Budget Revenue Per Available Room":
                    safe_div(
                        budget_revenue,
                        capacity
                    )
            })

    return pd.DataFrame(rows)


# =========================================================
# DATA PREPARATION
# =========================================================

def clean_raw_data(uploaded_file):
    df = pd.read_excel(uploaded_file)

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    required_cols = [
        "Stay Date",
        "Booking Date",
        "Market Segment",
        "LOS",
        "Booking Window",
        "Current Room Nights",
        "Current Room Revenue",
        "Current Room ADR"
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: "
            f"{missing_cols}"
        )

    optional_cols = [
        "Stay Year",
        "Stay Month",
        "Weekday Vs Weekend",
        "Stay Days of Week",
        "Property Code",
        "Market Category",
        "Market Prefix/Mini Hotel",
        "Rate Program Code",
        "Rate Program",
        "Room Pool",
        "Channel Indicator",
        "Channel Aggregate",
        "Channel Details",
        "Channel Type",
        "Channel Site",
        "Channel Partner",
        "Intermediary ID",
        "Travel Agency Name",
        "Loyalty Level",
        "Member Rate",
        "Source Country/Region",
        "D/I/R Indicator",
        "Trip Purpose"
    ]

    selected_cols = (
        required_cols +
        [
            col
            for col in optional_cols
            if col in df.columns
        ]
    )

    df = df[selected_cols].copy()

    df["Stay Date"] = (
        pd.to_datetime(
            df["Stay Date"],
            errors="coerce",
            dayfirst=False
        )
        .dt.normalize()
    )

    df["Booking Date"] = (
        pd.to_datetime(
            df["Booking Date"],
            errors="coerce",
            dayfirst=False
        )
        .dt.normalize()
    )

    numeric_cols = [
        "LOS",
        "Booking Window",
        "Current Room Nights",
        "Current Room Revenue",
        "Current Room ADR"
    ]

    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.strip()
        )

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    object_cols = (
        df.select_dtypes(
            include=["object"]
        ).columns
    )

    for col in object_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace(
                [
                    "nan",
                    "NaN",
                    "None",
                    "N/A",
                    ""
                ],
                "Unknown"
            )
        )

    fallback_category_cols = [
        "Market Category",
        "Market Prefix/Mini Hotel",
        "Rate Program Code",
        "Rate Program",
        "Room Pool",
        "Channel Aggregate",
        "Channel Details",
        "Channel Partner",
        "Travel Agency Name"
    ]

    for col in fallback_category_cols:
        if col not in df.columns:
            df[col] = "Unknown"

    df = df.dropna(
        subset=[
            "Stay Date",
            "Booking Date",
            "Market Segment",
            "LOS",
            "Booking Window",
            "Current Room Nights",
            "Current Room Revenue"
        ]
    )

    df = df[
        df["Market Segment"] !=
        "Complimentary"
    ]

    df = df[
        (df["Current Room Revenue"] > 0) &
        (df["Current Room Nights"] > 0)
    ]

    df = (
        df
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    df["Year"] = (
        df["Stay Date"].dt.year
    )

    df["Month"] = (
        df["Stay Date"].dt.month
    )

    df["Day"] = (
        df["Stay Date"].dt.day
    )

    df["YearMonth"] = (
        df["Stay Date"]
        .dt.to_period("M")
        .astype(str)
    )

    df["Quarter"] = (
        df["Stay Date"].dt.quarter
    )

    df["Day Of Week Number"] = (
        df["Stay Date"].dt.dayofweek
    )

    df["Is Weekend"] = (
        df["Day Of Week Number"]
        .isin([5, 6])
        .astype(int)
    )

    return df


def build_daily_df(df):
    if df.empty:
        return pd.DataFrame()

    daily = (
        df.groupby("Stay Date")
        .agg({
            "Current Room Revenue": "sum",
            "Current Room Nights": "sum",
            "LOS": "mean",
            "Booking Window": "mean",
            "Is Weekend": "max",
            "Quarter": "max",
            "Month": "max",
            "Year": "max",
            "Day Of Week Number": "max"
        })
        .reset_index()
    )

    segment = pd.pivot_table(
        df,
        index="Stay Date",
        columns="Market Segment",
        values="Current Room Nights",
        aggfunc="sum",
        fill_value=0
    )

    segment_rn = segment.copy()

    segment_rn.columns = [
        f"Segment_{col}_RN"
        for col in segment_rn.columns
    ]

    segment_rooms_sold = segment.copy()

    segment_rooms_sold.columns = [
        f"Segment_{col}_Rooms Sold"
        for col in segment_rooms_sold.columns
    ]

    daily = daily.merge(
        segment_rn.reset_index(),
        on="Stay Date",
        how="left"
    )

    daily = daily.merge(
        segment_rooms_sold.reset_index(),
        on="Stay Date",
        how="left"
    )

    daily = (
        daily
        .sort_values("Stay Date")
        .reset_index(drop=True)
    )

    daily["Day"] = (
        daily["Stay Date"].dt.day
    )

    daily["YearMonth"] = (
        daily["Stay Date"]
        .dt.to_period("M")
        .astype(str)
    )

    daily["Is Public Holiday"] = 0
    daily["Is Long Weekend"] = 0

    daily["Revenue MA7"] = (
        daily["Current Room Revenue"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    expanding_revenue_mean = (
        daily["Current Room Revenue"]
        .expanding()
        .mean()
    )

    daily["Revenue MA7"] = (
        daily["Revenue MA7"]
        .fillna(expanding_revenue_mean)
    )

    daily[
        "Revenue Moving Average 7 Days"
    ] = daily["Revenue MA7"]

    daily["Daily ADR"] = (
        daily["Current Room Revenue"] /
        daily["Current Room Nights"]
        .replace(0, np.nan)
    )

    daily[
        "Daily Average Room Rate"
    ] = daily["Daily ADR"]

    return (
        daily
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )


# =========================================================
# ACTIVE PERIOD AND CUTOFF LOGIC
# =========================================================

def split_data_by_active_period(
    df,
    active_year,
    active_month,
    cutoff_date,
    forecast_horizon_months
):
    cutoff_date = pd.to_datetime(
        cutoff_date
    ).normalize()

    active_start = pd.Timestamp(
        int(active_year),
        int(active_month),
        1
    )

    training_df = df[
        df["Stay Date"] < active_start
    ].copy()

    cutoff_exclusive = (
        cutoff_date +
        pd.Timedelta(days=1)
    )

    as_of_df = df[
        df["Booking Date"] <
        cutoff_exclusive
    ].copy()

    periods = []

    for offset in range(
        int(forecast_horizon_months)
    ):
        month_start = (
            active_start +
            pd.DateOffset(months=offset)
        )

        month_end = (
            month_start +
            pd.offsets.MonthEnd(0)
        )

        periods.append({
            "Period Type": (
                "Current Month"
                if offset == 0
                else f"Month + {offset}"
            ),
            "Year": int(month_start.year),
            "Month": int(month_start.month),
            "Month Start": month_start,
            "Month End": month_end
        })

    return (
        training_df,
        as_of_df,
        periods
    )


def calculate_period_as_of_metrics(
    df,
    target_year,
    target_month,
    cutoff_date
):
    data = df.copy()

    data["Stay Date"] = (
        pd.to_datetime(
            data["Stay Date"],
            errors="coerce"
        )
        .dt.normalize()
    )

    data["Booking Date"] = (
        pd.to_datetime(
            data["Booking Date"],
            errors="coerce"
        )
        .dt.normalize()
    )

    cutoff_date = pd.to_datetime(
        cutoff_date
    ).normalize()

    cutoff_exclusive = (
        cutoff_date +
        pd.Timedelta(days=1)
    )

    month_start = pd.Timestamp(
        int(target_year),
        int(target_month),
        1
    )

    month_end = (
        month_start +
        pd.offsets.MonthEnd(0)
    )

    month_end_exclusive = (
        month_end +
        pd.Timedelta(days=1)
    )

    booking_mask = (
        data["Booking Date"] <
        cutoff_exclusive
    )

    stay_month_mask = (
        (data["Stay Date"] >= month_start) &
        (
            data["Stay Date"] <
            month_end_exclusive
        )
    )

    known_df = data[
        booking_mask &
        stay_month_mask
    ].copy()

    known_revenue = (
        known_df[
            "Current Room Revenue"
        ].sum()
        if not known_df.empty
        else 0
    )

    known_rooms_sold = (
        known_df[
            "Current Room Nights"
        ].sum()
        if not known_df.empty
        else 0
    )

    known_average_room_rate = safe_div(
        known_revenue,
        known_rooms_sold
    )

    if cutoff_date < month_start:
        actual_running_df = (
            data.iloc[0:0].copy()
        )

    else:
        actual_end_date = min(
            cutoff_date,
            month_end
        )

        actual_end_exclusive = (
            actual_end_date +
            pd.Timedelta(days=1)
        )

        actual_running_df = data[
            booking_mask &
            stay_month_mask &
            (
                data["Stay Date"] <
                actual_end_exclusive
            )
        ].copy()

    actual_running_revenue = (
        actual_running_df[
            "Current Room Revenue"
        ].sum()
        if not actual_running_df.empty
        else 0
    )

    actual_running_rooms_sold = (
        actual_running_df[
            "Current Room Nights"
        ].sum()
        if not actual_running_df.empty
        else 0
    )

    actual_running_average_room_rate = (
        safe_div(
            actual_running_revenue,
            actual_running_rooms_sold
        )
    )

    if cutoff_date < month_start:
        future_start = month_start

    elif cutoff_date <= month_end:
        future_start = (
            cutoff_date +
            pd.Timedelta(days=1)
        )

    else:
        future_start = (
            month_end +
            pd.Timedelta(days=1)
        )

    known_future_df = known_df[
        known_df["Stay Date"] >=
        future_start
    ].copy()

    known_future_revenue = (
        known_future_df[
            "Current Room Revenue"
        ].sum()
        if not known_future_df.empty
        else 0
    )

    known_future_rooms_sold = (
        known_future_df[
            "Current Room Nights"
        ].sum()
        if not known_future_df.empty
        else 0
    )

    if not known_future_df.empty:
        known_future_daily = (
            known_future_df
            .groupby("Stay Date")[
                "Current Room Nights"
            ]
            .sum()
        )

        known_future_by_date = {
            pd.Timestamp(date).normalize():
                float(value)
            for date, value
            in known_future_daily.items()
        }

    else:
        known_future_by_date = {}

    return {
        "Known Revenue":
            float(known_revenue),

        "Known Rooms Sold":
            float(known_rooms_sold),

        "Known Average Room Rate":
            float(known_average_room_rate),

        "Known Booking Rows":
            int(len(known_df)),

        "Known Future Revenue":
            float(known_future_revenue),

        "Known Future Rooms Sold":
            float(known_future_rooms_sold),

        "Known Future Rooms by Date":
            known_future_by_date,

        "Actual Running Revenue":
            float(actual_running_revenue),

        "Actual Running Rooms Sold":
            float(actual_running_rooms_sold),

        "Actual Running Average Room Rate":
            float(
                actual_running_average_room_rate
            ),

        "Actual Running Rows":
            int(len(actual_running_df))
    }


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def sum_existing_columns(data, cols):
    existing_cols = [
        col
        for col in cols
        if col in data.columns
    ]

    if not existing_cols:
        return pd.Series(
            0,
            index=data.index
        )

    return data[
        existing_cols
    ].sum(axis=1)


def add_business_segment_features(data):
    temp = data.copy()

    temp["Group_RN"] = sum_existing_columns(
        temp,
        [
            "Segment_C Group Corporate_RN",
            "Segment_G Group Government_RN",
            "Segment_N Group Association_RN",
            "Segment_O Group Other_RN",
            "Segment_P Group Sports_RN",
            "Segment_S Group Social_RN",
            "Segment_T Group Tour Series/Wholesale_RN"
        ]
    )

    temp["Corporate_RN"] = (
        sum_existing_columns(
            temp,
            [
                "Segment_Special Corp_RN"
            ]
        )
    )

    temp["Gov_RN"] = (
        sum_existing_columns(
            temp,
            [
                "Segment_Govt/Military_RN"
            ]
        )
    )

    temp["CorpGov_RN"] = (
        temp["Corporate_RN"] +
        temp["Gov_RN"]
    )

    temp["Retail_RN"] = (
        sum_existing_columns(
            temp,
            [
                "Segment_Retail_RN",
                "Segment_AAA_RN",
                "Segment_Associate Leisure_RN",
                "Segment_Bonvoy Redemption_RN"
            ]
        )
    )

    temp["Package_RN"] = (
        sum_existing_columns(
            temp,
            [
                "Segment_Packages_RN"
            ]
        )
    )

    temp["Discount_RN"] = (
        sum_existing_columns(
            temp,
            [
                "Segment_Other Discount_RN"
            ]
        )
    )

    temp["Wholesale_RN"] = (
        sum_existing_columns(
            temp,
            [
                "Segment_Wholesaler_RN"
            ]
        )
    )

    temp["Transient_RN"] = (
        temp["CorpGov_RN"] +
        temp["Retail_RN"] +
        temp["Package_RN"] +
        temp["Discount_RN"] +
        temp["Wholesale_RN"]
    )

    temp["Group Rooms Sold"] = (
        temp["Group_RN"]
    )

    temp[
        "Corporate Government Rooms Sold"
    ] = temp["CorpGov_RN"]

    temp["Retail Rooms Sold"] = (
        temp["Retail_RN"] +
        temp["Package_RN"] +
        temp["Discount_RN"] +
        temp["Wholesale_RN"]
    )

    temp["Transient Rooms Sold"] = (
        temp["Transient_RN"]
    )

    return temp


def ensure_feature_columns(
    df,
    feature_list
):
    output = df.copy()

    for col in feature_list:
        if col not in output.columns:
            output[col] = 0

    return (
        output[feature_list]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )


# =========================================================
# AVERAGE ROOM RATE CAP
# =========================================================

def apply_average_rate_sanity_cap(
    projected_revenue,
    projected_rooms_sold,
    budget_average_room_rate,
    historical_daily_df,
    cap_multiplier=1.15,
    max_hist_to_budget_ratio=1.50
):
    if projected_rooms_sold <= 0:
        return {
            "capped_projected_revenue": 0,
            "raw_average_room_rate_before_cap": 0,
            "final_average_room_rate": 0,
            "average_room_rate_cap": 0,
            "average_room_rate_cap_applied": False,
            "cap_basis": "No rooms sold"
        }

    raw_rate = (
        projected_revenue /
        projected_rooms_sold
    )

    hist_p75 = 0
    hist_p90 = 0

    if (
        historical_daily_df is not None and
        not historical_daily_df.empty
    ):
        if (
            "Daily ADR"
            in historical_daily_df.columns
        ):
            rate_series = (
                historical_daily_df[
                    "Daily ADR"
                ].copy()
            )

        elif (
            "Daily Average Room Rate"
            in historical_daily_df.columns
        ):
            rate_series = (
                historical_daily_df[
                    "Daily Average Room Rate"
                ].copy()
            )

        else:
            rate_series = pd.Series(
                dtype=float
            )

        rate_series = (
            pd.to_numeric(
                rate_series,
                errors="coerce"
            )
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )

        rate_series = rate_series[
            rate_series > 0
        ]

        if not rate_series.empty:
            hist_p75 = (
                rate_series.quantile(0.75)
            )

            hist_p90 = (
                rate_series.quantile(0.90)
            )

    budget_rate = (
        float(budget_average_room_rate)
        if budget_average_room_rate > 0
        else 0
    )

    if (
        budget_rate > 0 and
        hist_p90 >
        (
            budget_rate *
            max_hist_to_budget_ratio
        )
    ):
        base_cap_rate = budget_rate

        basis = (
            "Budget Average Room Rate "
            "because historical P90 is too high"
        )

    else:
        base_cap_rate = max(
            budget_rate,
            hist_p75,
            hist_p90
        )

        basis = (
            "Max of budget and historical rate"
        )

    if base_cap_rate <= 0:
        return {
            "capped_projected_revenue":
                projected_revenue,

            "raw_average_room_rate_before_cap":
                raw_rate,

            "final_average_room_rate":
                raw_rate,

            "average_room_rate_cap":
                raw_rate,

            "average_room_rate_cap_applied":
                False,

            "cap_basis":
                "No valid cap basis"
        }

    average_room_rate_cap = (
        base_cap_rate *
        cap_multiplier
    )

    cap_applied = (
        raw_rate >
        average_room_rate_cap
    )

    if cap_applied:
        capped_revenue = (
            projected_rooms_sold *
            average_room_rate_cap
        )

    else:
        capped_revenue = (
            projected_revenue
        )

    return {
        "capped_projected_revenue":
            capped_revenue,

        "raw_average_room_rate_before_cap":
            raw_rate,

        "final_average_room_rate":
            safe_div(
                capped_revenue,
                projected_rooms_sold
            ),

        "average_room_rate_cap":
            average_room_rate_cap,

        "average_room_rate_cap_applied":
            cap_applied,

        "cap_basis":
            basis
    }


# =========================================================
# FORECAST SNAPSHOT
# =========================================================

def build_forecast_snapshot(
    df,
    period,
    budget_df,
    cutoff_date,
    total_rooms
):
    year = int(period["Year"])
    month = int(period["Month"])

    month_start = pd.to_datetime(
        period["Month Start"]
    )

    month_end = pd.to_datetime(
        period["Month End"]
    )

    cutoff_date = pd.to_datetime(
        cutoff_date
    ).normalize()

    days = int(month_end.day)

    monthly_capacity = (
        total_rooms *
        days
    )

    budget_row = budget_df[
        (budget_df["Year"] == year) &
        (budget_df["Month"] == month)
    ].copy()

    if budget_row.empty:
        raise ValueError(
            f"Budget tidak tersedia untuk "
            f"{year}-{month:02d}. "
            "Tambahkan budget pada build_budget_df()."
        )

    budget_row = budget_row.iloc[0]

    period_metrics = (
        calculate_period_as_of_metrics(
            df=df,
            target_year=year,
            target_month=month,
            cutoff_date=cutoff_date
        )
    )

    if cutoff_date < month_start:
        elapsed_days = 0
        future_start = month_start

    elif cutoff_date > month_end:
        elapsed_days = days

        future_start = (
            month_end +
            pd.Timedelta(days=1)
        )

    else:
        elapsed_days = int(
            cutoff_date.day
        )

        future_start = (
            cutoff_date +
            pd.Timedelta(days=1)
        )

    remaining_days = max(
        days - elapsed_days,
        0
    )

    remaining_room_capacity = max(
        monthly_capacity -
        period_metrics["Known Rooms Sold"],
        0
    )

    future_calendar_capacity = (
        total_rooms *
        remaining_days
    )

    remaining_future_room_capacity = max(
        future_calendar_capacity -
        period_metrics[
            "Known Future Rooms Sold"
        ],
        0
    )

    return {
        "Forecast Period":
            f"{year}-{month:02d}",

        "Period Type":
            period["Period Type"],

        "Year":
            year,

        "Month":
            month,

        "Month Start":
            month_start,

        "Month End":
            month_end,

        "Future Start Date":
            future_start,

        "Days In Month":
            days,

        "Elapsed Days":
            elapsed_days,

        "Remaining Days":
            remaining_days,

        "Total Rooms / Hotel Inventory":
            total_rooms,

        "Monthly Room Capacity":
            monthly_capacity,

        "Known Revenue":
            period_metrics[
                "Known Revenue"
            ],

        "Known Rooms Sold":
            period_metrics[
                "Known Rooms Sold"
            ],

        "Known Average Room Rate":
            period_metrics[
                "Known Average Room Rate"
            ],

        "Known Booking Rows":
            period_metrics[
                "Known Booking Rows"
            ],

        "Known Future Revenue":
            period_metrics[
                "Known Future Revenue"
            ],

        "Known Future Rooms Sold":
            period_metrics[
                "Known Future Rooms Sold"
            ],

        "Known Future Rooms by Date":
            period_metrics[
                "Known Future Rooms by Date"
            ],

        "Actual Running Revenue":
            period_metrics[
                "Actual Running Revenue"
            ],

        "Actual Running Rooms Sold":
            period_metrics[
                "Actual Running Rooms Sold"
            ],

        "Actual Running Average Room Rate":
            period_metrics[
                "Actual Running Average Room Rate"
            ],

        "Actual Running Rows":
            period_metrics[
                "Actual Running Rows"
            ],

        "Remaining Room Capacity":
            remaining_room_capacity,

        "Remaining Future Room Capacity":
            remaining_future_room_capacity,

        "Budget Revenue":
            budget_row[
                "Budget Revenue"
            ],

        "Budget Rooms Sold":
            budget_row[
                "Budget Rooms Sold"
            ],

        "Budget Average Room Rate":
            budget_row[
                "Budget Average Room Rate"
            ],

        "Budget Occupancy %":
            budget_row[
                "Budget Occupancy %"
            ],

        "Budget Revenue Per Available Room":
            budget_row[
                "Budget Revenue Per Available Room"
            ]
    }


# =========================================================
# FUTURE DAILY INPUT
# =========================================================

def build_future_rows(
    training_daily,
    snapshot,
    total_rooms,
    linear_features
):
    if snapshot["Remaining Days"] <= 0:
        return pd.DataFrame()

    future_dates = pd.date_range(
        start=snapshot[
            "Future Start Date"
        ],
        end=snapshot[
            "Month End"
        ],
        freq="D"
    )

    if len(future_dates) == 0:
        return pd.DataFrame()

    historical_same_month = (
        training_daily[
            training_daily["Month"] ==
            snapshot["Month"]
        ]
        .copy()
    )

    if historical_same_month.empty:
        historical_same_month = (
            training_daily.copy()
        )

    if historical_same_month.empty:
        raise ValueError(
            "Historical daily data tidak tersedia."
        )

    if snapshot["Elapsed Days"] > 0:
        raw_daily_rooms = safe_div(
            snapshot[
                "Actual Running Rooms Sold"
            ],
            snapshot[
                "Elapsed Days"
            ]
        )

        raw_daily_revenue = safe_div(
            snapshot[
                "Actual Running Revenue"
            ],
            snapshot[
                "Elapsed Days"
            ]
        )

    else:
        raw_daily_rooms = (
            historical_same_month[
                "Current Room Nights"
            ].median()
        )

        raw_daily_revenue = (
            historical_same_month[
                "Current Room Revenue"
            ].median()
        )

    if pd.isna(raw_daily_rooms):
        raw_daily_rooms = 0

    if pd.isna(raw_daily_revenue):
        raw_daily_revenue = 0

    raw_daily_rooms = float(
        raw_daily_rooms
    )

    raw_daily_revenue = float(
        raw_daily_revenue
    )

    segment_cols = [
        col
        for col in historical_same_month.columns
        if (
            col.startswith("Segment_") and
            col.endswith("_RN")
        )
    ]

    segment_medians = {
        col: (
            historical_same_month[
                col
            ].median()
            if col in historical_same_month.columns
            else 0
        )
        for col in segment_cols
    }

    total_segment_median = sum(
        float(value)
        for value in segment_medians.values()
        if pd.notna(value)
    )

    known_future_by_date = (
        snapshot.get(
            "Known Future Rooms by Date",
            {}
        )
    )

    rows = []

    for stay_date in future_dates:
        normalized_stay_date = (
            pd.Timestamp(
                stay_date
            ).normalize()
        )

        known_rooms_for_date = float(
            known_future_by_date.get(
                normalized_stay_date,
                0
            )
        )

        daily_rooms = max(
            raw_daily_rooms,
            known_rooms_for_date
        )

        daily_rooms = min(
            daily_rooms,
            total_rooms
        )

        daily_rooms = max(
            daily_rooms,
            0
        )

        if total_segment_median > 0:
            segment_daily = {
                col: (
                    float(
                        segment_medians[col]
                    ) /
                    total_segment_median *
                    daily_rooms
                )
                for col in segment_cols
            }

        else:
            segment_daily = {
                col: 0
                for col in segment_cols
            }

        row = {
            "Forecast Period":
                snapshot["Forecast Period"],

            "Period Type":
                snapshot["Period Type"],

            "Stay Date":
                stay_date,

            "Current Room Nights":
                daily_rooms,

            "LOS":
                historical_same_month[
                    "LOS"
                ].median(),

            "Booking Window":
                historical_same_month[
                    "Booking Window"
                ].median(),

            "Is Weekend":
                (
                    1
                    if stay_date.dayofweek
                    in [5, 6]
                    else 0
                ),

            "Quarter":
                stay_date.quarter,

            "Month":
                stay_date.month,

            "Day Of Week Number":
                stay_date.dayofweek,

            "Is Public Holiday":
                0,

            "Is Long Weekend":
                0,

            "Revenue MA7":
                raw_daily_revenue,

            "Revenue Moving Average 7 Days":
                raw_daily_revenue,

            "Known Rooms Sold for Date":
                known_rooms_for_date
        }

        for col in segment_cols:
            row[col] = (
                segment_daily.get(
                    col,
                    0
                )
            )

        rows.append(row)

    future_df = pd.DataFrame(rows)

    future_df = (
        add_business_segment_features(
            future_df
        )
    )

    for col in linear_features:
        if col not in future_df.columns:
            future_df[col] = 0

    return (
        future_df
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )


# =========================================================
# MODEL PREDICTION
# =========================================================

def predict_future_revenue(
    future_df,
    linear_model,
    linear_scaler,
    linear_features,
    naive_model,
    naive_scaler,
    naive_features,
    bin_reference
):
    if future_df.empty:
        return (
            0,
            0,
            0,
            0,
            future_df,
            pd.DataFrame()
        )

    x_linear = ensure_feature_columns(
        future_df,
        linear_features
    )

    linear_scaled = (
        linear_scaler.transform(
            x_linear
        )
    )

    future_df[
        "Forecast Revenue from Linear Regression"
    ] = np.maximum(
        linear_model.predict(
            linear_scaled
        ),
        0
    )

    linear_remaining = (
        future_df[
            "Forecast Revenue from Linear Regression"
        ].sum()
    )

    future_nb = (
        add_business_segment_features(
            future_df.copy()
        )
    )

    x_nb = ensure_feature_columns(
        future_nb,
        naive_features
    )

    nb_scaled = (
        naive_scaler.transform(
            x_nb
        )
    )

    probabilities = (
        naive_model.predict_proba(
            nb_scaled
        )
    )

    expected_revenue_list = []
    detail_rows = []

    for index, probability in enumerate(
        probabilities
    ):
        probability_df = pd.DataFrame({
            "Revenue State":
                naive_model.classes_,

            "Probability":
                probability
        })

        probability_df = (
            probability_df.merge(
                bin_reference,
                on="Revenue State",
                how="left"
            )
        )

        expected_revenue = (
            probability_df["Probability"] *
            probability_df[
                "Average Revenue in State"
            ]
        ).sum()

        expected_revenue_list.append(
            expected_revenue
        )

        best_row = probability_df.loc[
            probability_df[
                "Probability"
            ].idxmax()
        ]

        detail_rows.append({
            "Forecast Period":
                future_df.iloc[index][
                    "Forecast Period"
                ],

            "Period Type":
                future_df.iloc[index][
                    "Period Type"
                ],

            "Stay Date":
                future_df.iloc[index][
                    "Stay Date"
                ],

            "Expected Revenue from Naive Bayes":
                expected_revenue,

            "Most Likely Revenue State":
                best_row[
                    "Revenue State"
                ],

            "Revenue State Confidence %":
                best_row[
                    "Probability"
                ] * 100,

            "Minimum Revenue in State":
                best_row.get(
                    "Minimum Revenue in State",
                    0
                ),

            "Maximum Revenue in State":
                best_row.get(
                    "Maximum Revenue in State",
                    0
                ),

            "Average Revenue in State":
                best_row[
                    "Average Revenue in State"
                ]
        })

    future_df[
        "Forecast Revenue from Naive Bayes"
    ] = expected_revenue_list

    naive_remaining = (
        future_df[
            "Forecast Revenue from Naive Bayes"
        ].sum()
    )

    remaining_rooms = (
        future_df[
            "Current Room Nights"
        ].sum()
    )

    confidence = (
        np.max(
            probabilities,
            axis=1
        ).mean() *
        100
    )

    return (
        linear_remaining,
        naive_remaining,
        remaining_rooms,
        confidence,
        future_df,
        pd.DataFrame(detail_rows)
    )


# =========================================================
# FORECAST CORE
# =========================================================

def run_forecast(
    df,
    budget_df,
    linear_model,
    linear_scaler,
    linear_features,
    naive_model,
    naive_scaler,
    naive_features,
    bin_reference,
    active_year,
    active_month,
    cutoff_date,
    total_rooms,
    forecast_horizon_months
):
    (
        training_df,
        _,
        periods
    ) = split_data_by_active_period(
        df=df,
        active_year=active_year,
        active_month=active_month,
        cutoff_date=cutoff_date,
        forecast_horizon_months=
            forecast_horizon_months
    )

    training_daily = build_daily_df(
        training_df
    )

    if training_daily.empty:
        raise ValueError(
            "Training historical data kosong. "
            "Pastikan terdapat data sebelum bulan aktif."
        )

    training_daily = (
        add_business_segment_features(
            training_daily
        )
    )

    summary_rows = []
    future_rows = []
    nb_rows = []

    for period in periods:
        snapshot = build_forecast_snapshot(
            df=df,
            period=period,
            budget_df=budget_df,
            cutoff_date=cutoff_date,
            total_rooms=total_rooms
        )

        future_df = build_future_rows(
            training_daily=training_daily,
            snapshot=snapshot,
            total_rooms=total_rooms,
            linear_features=linear_features
        )

        (
            linear_remaining,
            naive_remaining,
            remaining_rooms,
            confidence,
            future_df,
            nb_detail
        ) = predict_future_revenue(
            future_df=future_df,
            linear_model=linear_model,
            linear_scaler=linear_scaler,
            linear_features=linear_features,
            naive_model=naive_model,
            naive_scaler=naive_scaler,
            naive_features=naive_features,
            bin_reference=bin_reference
        )

        model_linear_total = (
            snapshot[
                "Actual Running Revenue"
            ] +
            linear_remaining
        )

        model_naive_total_before_cap = (
            snapshot[
                "Actual Running Revenue"
            ] +
            naive_remaining
        )

        raw_linear = max(
            model_linear_total,
            snapshot["Known Revenue"]
        )

        raw_naive_before_cap = max(
            model_naive_total_before_cap,
            snapshot["Known Revenue"]
        )

        linear_otb_floor_applied = (
            model_linear_total <
            snapshot["Known Revenue"]
        )

        naive_otb_floor_applied = (
            model_naive_total_before_cap <
            snapshot["Known Revenue"]
        )

        projected_rooms = min(
            max(
                (
                    snapshot[
                        "Actual Running Rooms Sold"
                    ] +
                    remaining_rooms
                ),
                snapshot[
                    "Known Rooms Sold"
                ]
            ),
            snapshot[
                "Monthly Room Capacity"
            ]
        )

        cap_result = (
            apply_average_rate_sanity_cap(
                projected_revenue=
                    raw_naive_before_cap,

                projected_rooms_sold=
                    projected_rooms,

                budget_average_room_rate=
                    snapshot[
                        "Budget Average Room Rate"
                    ],

                historical_daily_df=
                    training_daily
            )
        )

        capped_naive = max(
            cap_result[
                "capped_projected_revenue"
            ],
            snapshot["Known Revenue"]
        )

        otb_floor_overrode_cap = (
            snapshot["Known Revenue"] >
            cap_result[
                "capped_projected_revenue"
            ]
        )

        projected_average_room_rate = (
            safe_div(
                capped_naive,
                projected_rooms
            )
        )

        revenue_gap = (
            snapshot["Budget Revenue"] -
            capped_naive
        )

        additional_rooms = 0

        if (
            revenue_gap > 0 and
            projected_average_room_rate > 0
        ):
            additional_rooms = int(
                np.ceil(
                    revenue_gap /
                    projected_average_room_rate
                )
            )

            additional_rooms = min(
                additional_rooms,
                int(
                    max(
                        snapshot[
                            "Monthly Room Capacity"
                        ] -
                        projected_rooms,
                        0
                    )
                )
            )

        if snapshot["Elapsed Days"] > 0:
            daily_rooms_pace_raw = (
                safe_div(
                    snapshot[
                        "Actual Running Rooms Sold"
                    ],
                    snapshot[
                        "Elapsed Days"
                    ]
                )
            )

        elif not future_df.empty:
            daily_rooms_pace_raw = (
                future_df[
                    "Current Room Nights"
                ].mean()
            )

        else:
            daily_rooms_pace_raw = 0

        daily_rooms_pace_capped = (
            future_df[
                "Current Room Nights"
            ].mean()
            if not future_df.empty
            else 0
        )

        cap_basis = (
            cap_result["cap_basis"]
        )

        if otb_floor_overrode_cap:
            cap_basis = (
                f"{cap_basis}; "
                "OTB floor retained"
            )

        summary_rows.append({
            "Forecast Period":
                snapshot[
                    "Forecast Period"
                ],

            "Period Type":
                snapshot[
                    "Period Type"
                ],

            "Cutoff Date":
                pd.to_datetime(
                    cutoff_date
                ).strftime("%Y-%m-%d"),

            "Total Rooms / Hotel Inventory":
                total_rooms,

            "Monthly Room Capacity":
                snapshot[
                    "Monthly Room Capacity"
                ],

            "Known Revenue":
                snapshot[
                    "Known Revenue"
                ],

            "OTB Revenue as of Cutoff":
                snapshot[
                    "Known Revenue"
                ],

            "Known Rooms Sold":
                snapshot[
                    "Known Rooms Sold"
                ],

            "OTB Rooms Sold as of Cutoff":
                snapshot[
                    "Known Rooms Sold"
                ],

            "Known Average Room Rate":
                snapshot[
                    "Known Average Room Rate"
                ],

            "Known Booking Rows":
                snapshot[
                    "Known Booking Rows"
                ],

            "Known Future Revenue":
                snapshot[
                    "Known Future Revenue"
                ],

            "Known Future Rooms Sold":
                snapshot[
                    "Known Future Rooms Sold"
                ],

            "Actual Running Revenue":
                snapshot[
                    "Actual Running Revenue"
                ],

            "Actual Running Rooms Sold":
                snapshot[
                    "Actual Running Rooms Sold"
                ],

            "Actual Running Average Room Rate":
                snapshot[
                    "Actual Running Average Room Rate"
                ],

            "Actual Running Rows":
                snapshot[
                    "Actual Running Rows"
                ],

            "Elapsed Days":
                snapshot[
                    "Elapsed Days"
                ],

            "Remaining Days":
                snapshot[
                    "Remaining Days"
                ],

            "Remaining Room Capacity":
                snapshot[
                    "Remaining Room Capacity"
                ],

            "Remaining Future Room Capacity":
                snapshot[
                    "Remaining Future Room Capacity"
                ],

            "Daily Rooms Sold Pace Raw":
                daily_rooms_pace_raw,

            "Daily Rooms Sold Pace Capped by Inventory":
                daily_rooms_pace_capped,

            "Model Projected Revenue from Linear Regression before OTB Floor":
                model_linear_total,

            "Model Projected Revenue from Naive Bayes before OTB Floor":
                model_naive_total_before_cap,

            "Linear Regression OTB Floor Applied":
                linear_otb_floor_applied,

            "Naive Bayes OTB Floor Applied":
                naive_otb_floor_applied,

            "Raw Projected Revenue from Linear Regression":
                raw_linear,

            "Raw Projected Revenue from Naive Bayes before Cap":
                raw_naive_before_cap,

            "Raw Average Room Rate before Cap":
                cap_result[
                    "raw_average_room_rate_before_cap"
                ],

            "Average Room Rate Cap":
                cap_result[
                    "average_room_rate_cap"
                ],

            "Average Room Rate Cap Applied":
                cap_result[
                    "average_room_rate_cap_applied"
                ],

            "Average Room Rate Cap Basis":
                cap_basis,

            "OTB Floor Overrode Average Room Rate Cap":
                otb_floor_overrode_cap,

            "Raw Projected Revenue from Naive Bayes":
                capped_naive,

            "Projected Revenue from Linear Regression":
                raw_linear,

            "Projected Revenue from Naive Bayes":
                capped_naive,

            "Demand Factor":
                1.00,

            "Budget Revenue":
                snapshot[
                    "Budget Revenue"
                ],

            "Revenue Gap":
                revenue_gap,

            "Revenue Achievement %":
                safe_div(
                    capped_naive,
                    snapshot[
                        "Budget Revenue"
                    ]
                ) * 100,

            "Projected Rooms Sold":
                projected_rooms,

            "Budget Rooms Sold":
                snapshot[
                    "Budget Rooms Sold"
                ],

            "Rooms Sold Gap":
                (
                    snapshot[
                        "Budget Rooms Sold"
                    ] -
                    projected_rooms
                ),

            "Rooms Sold Achievement %":
                safe_div(
                    projected_rooms,
                    snapshot[
                        "Budget Rooms Sold"
                    ]
                ) * 100,

            "Projected Average Room Rate":
                projected_average_room_rate,

            "Budget Average Room Rate":
                snapshot[
                    "Budget Average Room Rate"
                ],

            "Average Room Rate Gap":
                (
                    snapshot[
                        "Budget Average Room Rate"
                    ] -
                    projected_average_room_rate
                ),

            "Average Room Rate Achievement %":
                safe_div(
                    projected_average_room_rate,
                    snapshot[
                        "Budget Average Room Rate"
                    ]
                ) * 100,

            "Projected Occupancy %":
                safe_div(
                    projected_rooms,
                    snapshot[
                        "Monthly Room Capacity"
                    ]
                ) * 100,

            "Budget Occupancy %":
                snapshot[
                    "Budget Occupancy %"
                ],

            "Projected Revenue Per Available Room":
                safe_div(
                    capped_naive,
                    snapshot[
                        "Monthly Room Capacity"
                    ]
                ),

            "Budget Revenue Per Available Room":
                snapshot[
                    "Budget Revenue Per Available Room"
                ],

            "Additional Rooms Sold Needed":
                additional_rooms,

            "Naive Bayes Average Confidence %":
                confidence
        })

        if not future_df.empty:
            future_rows.append(
                future_df
            )

        if not nb_detail.empty:
            nb_rows.append(
                nb_detail
            )

    summary_df = pd.DataFrame(
        summary_rows
    )

    future_all_df = (
        pd.concat(
            future_rows,
            ignore_index=True
        )
        if future_rows
        else pd.DataFrame()
    )

    nb_all_df = (
        pd.concat(
            nb_rows,
            ignore_index=True
        )
        if nb_rows
        else pd.DataFrame()
    )

    return (
        summary_df,
        future_all_df,
        nb_all_df
    )


# =========================================================
# DEMAND SCENARIO
# =========================================================

def scenario_to_factor(scenario):
    mapping = {
        "Very Low Demand / Sacrifice": 0.80,
        "Low Demand": 0.85,
        "Conservative": 0.90,
        "Slightly Conservative": 0.95,
        "Normal": 1.00,
        "Slight Upside": 1.05,
        "High Demand / Event": 1.10,
        "Peak Demand": 1.20
    }

    return mapping.get(
        scenario,
        1.00
    )


def apply_demand_factor(
    forecast_raw,
    demand_factor_input
):
    forecast_summary = (
        forecast_raw.copy()
    )

    forecast_summary[
        "Demand Factor"
    ] = (
        forecast_summary[
            "Forecast Period"
        ]
        .map(
            demand_factor_input
        )
        .fillna(1.00)
    )

    known_revenue = (
        forecast_summary[
            "Known Revenue"
        ]
    )

    naive_future_component = (
        forecast_summary[
            "Raw Projected Revenue from Naive Bayes"
        ] -
        known_revenue
    ).clip(lower=0)

    linear_future_component = (
        forecast_summary[
            "Raw Projected Revenue from Linear Regression"
        ] -
        known_revenue
    ).clip(lower=0)

    forecast_summary[
        "Projected Revenue from Naive Bayes"
    ] = (
        known_revenue +
        (
            naive_future_component *
            forecast_summary[
                "Demand Factor"
            ]
        )
    )

    forecast_summary[
        "Projected Revenue from Linear Regression"
    ] = (
        known_revenue +
        (
            linear_future_component *
            forecast_summary[
                "Demand Factor"
            ]
        )
    )

    forecast_summary[
        "Projected Average Room Rate"
    ] = safe_div_series(
        forecast_summary[
            "Projected Revenue from Naive Bayes"
        ],
        forecast_summary[
            "Projected Rooms Sold"
        ]
    )

    forecast_summary[
        "Projected Revenue Per Available Room"
    ] = safe_div_series(
        forecast_summary[
            "Projected Revenue from Naive Bayes"
        ],
        forecast_summary[
            "Monthly Room Capacity"
        ]
    )

    forecast_summary[
        "Revenue Gap"
    ] = (
        forecast_summary[
            "Budget Revenue"
        ] -
        forecast_summary[
            "Projected Revenue from Naive Bayes"
        ]
    )

    forecast_summary[
        "Revenue Achievement %"
    ] = (
        safe_div_series(
            forecast_summary[
                "Projected Revenue from Naive Bayes"
            ],
            forecast_summary[
                "Budget Revenue"
            ]
        ) *
        100
    )

    forecast_summary[
        "Average Room Rate Gap"
    ] = (
        forecast_summary[
            "Budget Average Room Rate"
        ] -
        forecast_summary[
            "Projected Average Room Rate"
        ]
    )

    forecast_summary[
        "Average Room Rate Achievement %"
    ] = (
        safe_div_series(
            forecast_summary[
                "Projected Average Room Rate"
            ],
            forecast_summary[
                "Budget Average Room Rate"
            ]
        ) *
        100
    )

    def calculate_additional_rooms(row):
        if (
            row["Revenue Gap"] <= 0 or
            row[
                "Projected Average Room Rate"
            ] <= 0
        ):
            return 0

        required_rooms = int(
            np.ceil(
                row["Revenue Gap"] /
                row[
                    "Projected Average Room Rate"
                ]
            )
        )

        available_capacity = int(
            max(
                row[
                    "Monthly Room Capacity"
                ] -
                row[
                    "Projected Rooms Sold"
                ],
                0
            )
        )

        return min(
            required_rooms,
            available_capacity
        )

    forecast_summary[
        "Additional Rooms Sold Needed"
    ] = forecast_summary.apply(
        calculate_additional_rooms,
        axis=1
    )

    return (
        forecast_summary
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )


# =========================================================
# SEGMENT ADJUSTMENT
# =========================================================

def classify_adjustment_segment(row):
    combined = " ".join([
        str(
            row.get(col, "")
        ).upper()
        for col in [
            "Market Segment",
            "Market Category",
            "Rate Program",
            "Market Prefix/Mini Hotel",
            "Channel Aggregate",
            "Travel Agency Name"
        ]
    ])

    if any(
        keyword in combined
        for keyword in [
            "GROUP",
            "WED",
            "WEDDING",
            "SANGJIT",
            "ENGAGEMENT",
            "SOCIAL",
            "ASSOCIATION",
            "SPORTS",
            "TOUR SERIES"
        ]
    ):
        return "Group"

    if any(
        keyword in combined
        for keyword in [
            "CORP",
            "CORPORATE",
            "GOVT",
            "GOVERNMENT",
            "MILITARY",
            "KEMEN",
            "KEMLU",
            "LKPP",
            "LPDP",
            "BPOM",
            "BPKP",
            "EMBASSY",
            "KEDUTAAN",
            "PT ",
            " PT",
            "TBK",
            "BANK"
        ]
    ):
        return (
            "Transient - "
            "Corporate/Government"
        )

    if any(
        keyword in combined
        for keyword in [
            "PACKAGE",
            "BREAKFAST",
            "EXPERIENCE"
        ]
    ):
        return "Transient - Package"

    if any(
        keyword in combined
        for keyword in [
            "DISCOUNT",
            "PROMO",
            "AAA",
            "REDEMPTION",
            "BONVOY",
            "SALE",
            "DEAL"
        ]
    ):
        return "Transient - Discount"

    if any(
        keyword in combined
        for keyword in [
            "WHOLESALE",
            "WHOLESALER",
            "TOUR",
            "TRAVEL",
            "KHIRI",
            "TRAILS"
        ]
    ):
        return "Transient - Wholesale"

    return "Transient - Retail"


def get_adjustment_rate_benchmark(
    df,
    active_year,
    active_month
):
    historical_df = df[
        (
            df["Stay Date"].dt.year <
            active_year
        ) |
        (
            (
                df["Stay Date"].dt.year ==
                active_year
            ) &
            (
                df["Stay Date"].dt.month <
                active_month
            )
        )
    ].copy()

    if historical_df.empty:
        historical_df = df.copy()

    historical_df[
        "Adjustment Segment"
    ] = historical_df.apply(
        classify_adjustment_segment,
        axis=1
    )

    benchmark = (
        historical_df
        .groupby(
            "Adjustment Segment"
        )
        .agg({
            "Current Room Revenue": "sum",
            "Current Room Nights": "sum"
        })
        .reset_index()
    )

    benchmark[
        "Benchmark Average Room Rate"
    ] = (
        benchmark[
            "Current Room Revenue"
        ] /
        benchmark[
            "Current Room Nights"
        ].replace(0, np.nan)
    )

    benchmark[
        "Benchmark Average Room Rate"
    ] = (
        benchmark[
            "Benchmark Average Room Rate"
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    default_rate = safe_div(
        historical_df[
            "Current Room Revenue"
        ].sum(),
        historical_df[
            "Current Room Nights"
        ].sum()
    )

    rate_dict = dict(
        zip(
            benchmark[
                "Adjustment Segment"
            ],
            benchmark[
                "Benchmark Average Room Rate"
            ]
        )
    )

    adjustment_segments = [
        "Group",
        "Transient - Corporate/Government",
        "Transient - Retail",
        "Transient - Package",
        "Transient - Discount",
        "Transient - Wholesale"
    ]

    for segment in adjustment_segments:
        if (
            segment not in rate_dict or
            rate_dict[segment] == 0
        ):
            rate_dict[segment] = (
                default_rate
            )

    return (
        rate_dict,
        benchmark
    )


def apply_segment_adjustment(
    forecast_summary,
    adjustment_input,
    rate_benchmark
):
    rows = []

    for _, row in forecast_summary.iterrows():
        period = row[
            "Forecast Period"
        ]

        base_revenue = row[
            "Projected Revenue from Naive Bayes"
        ]

        base_rooms = row[
            "Projected Rooms Sold"
        ]

        capacity = row[
            "Monthly Room Capacity"
        ]

        added_rooms_total = 0
        added_revenue_total = 0

        for (
            segment,
            rooms_added
        ) in adjustment_input.get(
            period,
            {}
        ).items():
            rooms_added = float(
                rooms_added
            )

            segment_rate = (
                rate_benchmark.get(
                    segment,
                    row[
                        "Projected Average Room Rate"
                    ]
                )
            )

            added_rooms_total += (
                rooms_added
            )

            added_revenue_total += (
                rooms_added *
                segment_rate
            )

        adjusted_rooms = min(
            base_rooms +
            added_rooms_total,
            capacity
        )

        allowed_added_rooms = max(
            adjusted_rooms -
            base_rooms,
            0
        )

        if (
            added_rooms_total > 0 and
            allowed_added_rooms <
            added_rooms_total
        ):
            added_revenue_total = (
                added_revenue_total *
                (
                    allowed_added_rooms /
                    added_rooms_total
                )
            )

        adjusted_revenue = (
            base_revenue +
            added_revenue_total
        )

        adjusted_average_room_rate = (
            safe_div(
                adjusted_revenue,
                adjusted_rooms
            )
        )

        rows.append({
            "Forecast Period":
                period,

            "Period Type":
                row["Period Type"],

            "Base Projected Revenue":
                base_revenue,

            "Adjustment Rooms Sold":
                added_rooms_total,

            "Adjustment Revenue":
                added_revenue_total,

            "Adjusted Projected Revenue":
                adjusted_revenue,

            "Budget Revenue":
                row["Budget Revenue"],

            "Adjusted Revenue Gap":
                (
                    row["Budget Revenue"] -
                    adjusted_revenue
                ),

            "Adjusted Revenue Achievement %":
                safe_div(
                    adjusted_revenue,
                    row["Budget Revenue"]
                ) * 100,

            "Base Projected Rooms Sold":
                base_rooms,

            "Adjusted Projected Rooms Sold":
                adjusted_rooms,

            "Budget Rooms Sold":
                row["Budget Rooms Sold"],

            "Adjusted Rooms Sold Achievement %":
                safe_div(
                    adjusted_rooms,
                    row["Budget Rooms Sold"]
                ) * 100,

            "Adjusted Average Room Rate":
                adjusted_average_room_rate,

            "Budget Average Room Rate":
                row[
                    "Budget Average Room Rate"
                ],

            "Adjusted Average Room Rate Achievement %":
                safe_div(
                    adjusted_average_room_rate,
                    row[
                        "Budget Average Room Rate"
                    ]
                ) * 100,

            "Adjusted Occupancy %":
                safe_div(
                    adjusted_rooms,
                    capacity
                ) * 100,

            "Adjusted Revenue Per Available Room":
                safe_div(
                    adjusted_revenue,
                    capacity
                )
        })

    return pd.DataFrame(rows)


# =========================================================
# CHARTS
# =========================================================

def build_revenue_chart(
    forecast_summary,
    adjusted_forecast
):
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=forecast_summary[
                "Forecast Period"
            ],
            y=forecast_summary[
                "Projected Revenue from Naive Bayes"
            ],
            mode="lines+markers",
            name="Projected Revenue"
        )
    )

    figure.add_trace(
        go.Scatter(
            x=forecast_summary[
                "Forecast Period"
            ],
            y=forecast_summary[
                "Budget Revenue"
            ],
            mode="lines+markers",
            name="Budget Revenue"
        )
    )

    if (
        adjusted_forecast is not None and
        not adjusted_forecast.empty
    ):
        figure.add_trace(
            go.Scatter(
                x=adjusted_forecast[
                    "Forecast Period"
                ],
                y=adjusted_forecast[
                    "Adjusted Projected Revenue"
                ],
                mode="lines+markers",
                name=(
                    "Adjusted Projected "
                    "Revenue"
                )
            )
        )

    figure.update_layout(
        title=(
            "Projected Revenue vs "
            "Budget Revenue"
        ),
        xaxis_title=(
            "Forecast Period"
        ),
        yaxis_title="Revenue",
        hovermode="x unified"
    )

    return figure


def build_rooms_chart(
    forecast_summary,
    adjusted_forecast
):
    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=forecast_summary[
                "Forecast Period"
            ],
            y=forecast_summary[
                "Projected Rooms Sold"
            ],
            name="Projected Rooms Sold"
        )
    )

    figure.add_trace(
        go.Bar(
            x=forecast_summary[
                "Forecast Period"
            ],
            y=forecast_summary[
                "Budget Rooms Sold"
            ],
            name="Budget Rooms Sold"
        )
    )

    if (
        adjusted_forecast is not None and
        not adjusted_forecast.empty
    ):
        figure.add_trace(
            go.Bar(
                x=adjusted_forecast[
                    "Forecast Period"
                ],
                y=adjusted_forecast[
                    "Adjusted Projected Rooms Sold"
                ],
                name="Adjusted Rooms Sold"
            )
        )

    figure.update_layout(
        title=(
            "Projected Rooms Sold vs "
            "Budget Rooms Sold"
        ),
        xaxis_title="Forecast Period",
        yaxis_title="Rooms Sold",
        barmode="group",
        hovermode="x unified"
    )

    return figure


def build_achievement_chart(
    forecast_summary,
    adjusted_forecast
):
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=forecast_summary[
                "Forecast Period"
            ],
            y=forecast_summary[
                "Revenue Achievement %"
            ],
            mode="lines+markers",
            name=(
                "Baseline Revenue "
                "Achievement %"
            )
        )
    )

    if (
        adjusted_forecast is not None and
        not adjusted_forecast.empty
    ):
        figure.add_trace(
            go.Scatter(
                x=adjusted_forecast[
                    "Forecast Period"
                ],
                y=adjusted_forecast[
                    "Adjusted Revenue Achievement %"
                ],
                mode="lines+markers",
                name=(
                    "Adjusted Revenue "
                    "Achievement %"
                )
            )
        )

    figure.add_hline(
        y=100,
        line_dash="dash",
        annotation_text="Budget 100%"
    )

    figure.update_layout(
        title=(
            "Revenue Achievement "
            "Percentage"
        ),
        xaxis_title="Forecast Period",
        yaxis_title="Achievement %",
        hovermode="x unified"
    )

    return figure


# =========================================================
# REVENUE STATE SUMMARY
# =========================================================

def format_revenue_state_label(value):
    try:
        if pd.isna(value):
            return "-"

        text = format_state_value(
            value
        )

        if str(text).lower().startswith(
            "state"
        ):
            return text

        return f"State {text}"

    except Exception:
        return "-"


def add_dominant_revenue_state_to_summary(
    forecast_summary,
    nb_detail
):
    if (
        forecast_summary is None or
        forecast_summary.empty
    ):
        return forecast_summary

    output = forecast_summary.copy()

    if (
        nb_detail is None or
        nb_detail.empty
    ):
        output[
            "Dominant Revenue State"
        ] = "-"

        output[
            "Dominant Revenue State Confidence %"
        ] = 0

        return output

    required_cols = [
        "Forecast Period",
        "Most Likely Revenue State",
        "Revenue State Confidence %"
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in nb_detail.columns
    ]

    if missing_cols:
        output[
            "Dominant Revenue State"
        ] = "-"

        output[
            "Dominant Revenue State Confidence %"
        ] = 0

        return output

    state_summary = (
        nb_detail
        .groupby(
            [
                "Forecast Period",
                "Most Likely Revenue State"
            ],
            as_index=False
        )
        .agg(
            State_Days=(
                "Most Likely Revenue State",
                "count"
            ),
            Dominant_Revenue_State_Confidence=(
                "Revenue State Confidence %",
                "mean"
            )
        )
        .sort_values(
            [
                "Forecast Period",
                "State_Days",
                "Dominant_Revenue_State_Confidence"
            ],
            ascending=[
                True,
                False,
                False
            ]
        )
        .drop_duplicates(
            "Forecast Period"
        )
        .rename(
            columns={
                "Most Likely Revenue State":
                    "Dominant Revenue State",

                "Dominant_Revenue_State_Confidence":
                    "Dominant Revenue State Confidence %"
            }
        )
    )

    output = output.merge(
        state_summary[
            [
                "Forecast Period",
                "Dominant Revenue State",
                "Dominant Revenue State Confidence %",
                "State_Days"
            ]
        ],
        on="Forecast Period",
        how="left"
    )

    output[
        "Dominant Revenue State"
    ] = (
        output[
            "Dominant Revenue State"
        ]
        .fillna("-")
    )

    output[
        "Dominant Revenue State Confidence %"
    ] = (
        output[
            "Dominant Revenue State Confidence %"
        ]
        .fillna(0)
    )

    output[
        "State_Days"
    ] = (
        output[
            "State_Days"
        ]
        .fillna(0)
    )

    return output


# =========================================================
# SESSION STATE HELPERS
# =========================================================

adjustment_segments = [
    "Group",
    "Transient - Corporate/Government",
    "Transient - Retail",
    "Transient - Package",
    "Transient - Discount",
    "Transient - Wholesale"
]

scenario_options = [
    "Very Low Demand / Sacrifice",
    "Low Demand",
    "Conservative",
    "Slightly Conservative",
    "Normal",
    "Slight Upside",
    "High Demand / Event",
    "Peak Demand"
]


def build_default_demand_input(
    forecast_raw
):
    return {
        str(
            row["Forecast Period"]
        ): 1.00
        for _, row
        in forecast_raw.iterrows()
    }


def build_default_adjustment_input(
    forecast_summary,
    segment_list
):
    return {
        str(
            row["Forecast Period"]
        ): {
            segment: 0
            for segment in segment_list
        }
        for _, row
        in forecast_summary.iterrows()
    }


def factor_to_scenario_label(factor):
    factor = float(factor)

    mapping = {
        0.80:
            "Very Low Demand / Sacrifice",

        0.85:
            "Low Demand",

        0.90:
            "Conservative",

        0.95:
            "Slightly Conservative",

        1.00:
            "Normal",

        1.05:
            "Slight Upside",

        1.10:
            "High Demand / Event",

        1.20:
            "Peak Demand"
    }

    return mapping.get(
        factor,
        "Normal"
    )


def calculate_actual_running_current_month(
    df,
    active_year,
    active_month,
    cutoff_date
):
    metrics = (
        calculate_period_as_of_metrics(
            df=df,
            target_year=active_year,
            target_month=active_month,
            cutoff_date=cutoff_date
        )
    )

    return {
        "Actual Running Revenue":
            metrics[
                "Actual Running Revenue"
            ],

        "Actual Running Rooms Sold":
            metrics[
                "Actual Running Rooms Sold"
            ],

        "Actual Running Average Room Rate":
            metrics[
                "Actual Running Average Room Rate"
            ]
    }


# =========================================================
# PAGE HEADER
# =========================================================

st.title(
    "Hotel Revenue Forecasting Dashboard"
)

st.caption(
    "Deployment menggunakan saved model. "
    "Tidak ada training ulang di Streamlit."
)


# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================

if (
    "forecast_package"
    not in st.session_state
):
    st.session_state[
        "forecast_package"
    ] = None

if (
    "applied_demand_factor_input"
    not in st.session_state
):
    st.session_state[
        "applied_demand_factor_input"
    ] = {}

if (
    "applied_adjustment_input"
    not in st.session_state
):
    st.session_state[
        "applied_adjustment_input"
    ] = {}


# =========================================================
# INPUT FORM
# =========================================================

with st.sidebar:
    st.header("Input Forecast")

    with st.form(
        "run_forecast_form"
    ):
        uploaded_file = (
            st.file_uploader(
                "Upload Excel Data Hotel",
                type=["xlsx", "xls"]
            )
        )

        model_dir = st.text_input(
            "Folder Model",
            value="Model"
        )

        forecast_year = (
            st.number_input(
                "Tahun Bulan Aktif",
                min_value=2023,
                max_value=2030,
                value=2026,
                step=1
            )
        )

        forecast_month = (
            st.selectbox(
                "Bulan Aktif Prediksi",
                list(range(1, 13)),
                index=4,
                format_func=lambda month: (
                    pd.Timestamp(
                        2026,
                        month,
                        1
                    ).strftime("%B")
                )
            )
        )

        cutoff_day = (
            st.number_input(
                "Hari Cutoff Booking Date",
                min_value=1,
                max_value=31,
                value=29,
                step=1,
                help=(
                    "Tanggal cutoff otomatis "
                    "mengikuti tahun dan bulan aktif. "
                    "Contoh: bulan aktif Juni dan "
                    "hari cutoff 29 akan menjadi "
                    "29 Juni."
                )
            )
        )

        forecast_horizon_months = (
            st.number_input(
                "Jumlah Bulan Forecast",
                min_value=1,
                max_value=12,
                value=3,
                step=1
            )
        )

        run_button = (
            st.form_submit_button(
                "Run Forecast"
            )
        )


# =========================================================
# RUN FORECAST
# =========================================================

if run_button:
    if uploaded_file is None:
        st.error(
            "Upload file Excel hotel "
            "terlebih dahulu."
        )
        st.stop()

    st.session_state[
        "forecast_package"
    ] = None

    with st.spinner(
        "Loading saved model dan "
        "membuat forecast..."
    ):
        try:
            df = clean_raw_data(
                uploaded_file
            )

            active_year = int(
                forecast_year
            )

            active_month = int(
                forecast_month
            )

            days_in_active_month = (
                pd.Timestamp(
                    active_year,
                    active_month,
                    1
                ).days_in_month
            )

            resolved_cutoff_day = min(
                int(cutoff_day),
                days_in_active_month
            )

            resolved_cutoff_date = (
                pd.Timestamp(
                    active_year,
                    active_month,
                    resolved_cutoff_day
                )
            )

            total_rooms = (
                HOTEL_TOTAL_ROOMS
            )

            budget_df = build_budget_df(
                total_rooms
            )

            (
                training_preview,
                as_of_preview,
                _
            ) = split_data_by_active_period(
                df=df,
                active_year=active_year,
                active_month=active_month,
                cutoff_date=
                    resolved_cutoff_date,
                forecast_horizon_months=
                    int(
                        forecast_horizon_months
                    )
            )

            (
                linear_model,
                linear_scaler,
                linear_features,
                naive_model,
                naive_scaler,
                naive_features,
                bin_reference
            ) = load_saved_model_bundle(
                model_dir=model_dir
            )

            (
                forecast_raw,
                forecast_future,
                nb_detail
            ) = run_forecast(
                df=df,
                budget_df=budget_df,
                linear_model=linear_model,
                linear_scaler=linear_scaler,
                linear_features=linear_features,
                naive_model=naive_model,
                naive_scaler=naive_scaler,
                naive_features=naive_features,
                bin_reference=bin_reference,
                active_year=active_year,
                active_month=active_month,
                cutoff_date=
                    resolved_cutoff_date,
                total_rooms=total_rooms,
                forecast_horizon_months=
                    int(
                        forecast_horizon_months
                    )
            )

            (
                rate_benchmark,
                rate_benchmark_table
            ) = get_adjustment_rate_benchmark(
                df=df,
                active_year=active_year,
                active_month=active_month
            )

            actual_running = (
                calculate_actual_running_current_month(
                    df=df,
                    active_year=active_year,
                    active_month=active_month,
                    cutoff_date=
                        resolved_cutoff_date
                )
            )

            old_widget_keys = [
                key
                for key
                in list(
                    st.session_state.keys()
                )
                if (
                    key.startswith(
                        "demand_form_"
                    ) or
                    key.startswith(
                        "adjust_form_"
                    )
                )
            ]

            for key in old_widget_keys:
                del st.session_state[key]

            st.session_state[
                "forecast_package"
            ] = {
                "df":
                    df,

                "budget_df":
                    budget_df,

                "training_preview":
                    training_preview,

                "as_of_preview":
                    as_of_preview,

                "linear_features":
                    linear_features,

                "naive_features":
                    naive_features,

                "bin_reference":
                    bin_reference,

                "forecast_raw":
                    forecast_raw,

                "forecast_future":
                    forecast_future,

                "nb_detail":
                    nb_detail,

                "rate_benchmark":
                    rate_benchmark,

                "rate_benchmark_table":
                    rate_benchmark_table,

                "actual_running":
                    actual_running,

                "inputs": {
                    "forecast_year":
                        active_year,

                    "forecast_month":
                        active_month,

                    "cutoff_date":
                        resolved_cutoff_date,

                    "forecast_horizon_months":
                        int(
                            forecast_horizon_months
                        ),

                    "total_rooms":
                        total_rooms
                }
            }

            st.session_state[
                "applied_demand_factor_input"
            ] = build_default_demand_input(
                forecast_raw
            )

            temp_summary = (
                apply_demand_factor(
                    forecast_raw,
                    st.session_state[
                        "applied_demand_factor_input"
                    ]
                )
            )

            st.session_state[
                "applied_adjustment_input"
            ] = (
                build_default_adjustment_input(
                    temp_summary,
                    adjustment_segments
                )
            )

        except Exception as error:
            st.error(
                f"Forecast gagal dibuat: "
                f"{error}"
            )
            st.stop()


# =========================================================
# STOP BEFORE FORECAST
# =========================================================

if (
    st.session_state[
        "forecast_package"
    ] is None
):
    st.info(
        "Upload file, isi parameter, "
        "lalu klik Run Forecast."
    )
    st.stop()


# =========================================================
# LOAD PACKAGE
# =========================================================

package = (
    st.session_state[
        "forecast_package"
    ]
)

df = package["df"]
budget_df = package["budget_df"]
training_preview = package[
    "training_preview"
]
as_of_preview = package[
    "as_of_preview"
]
linear_features = package[
    "linear_features"
]
naive_features = package[
    "naive_features"
]
bin_reference = package[
    "bin_reference"
]
forecast_raw = package[
    "forecast_raw"
]
forecast_future = package[
    "forecast_future"
]
nb_detail = package[
    "nb_detail"
]
rate_benchmark = package[
    "rate_benchmark"
]
rate_benchmark_table = package[
    "rate_benchmark_table"
]
actual_running = package[
    "actual_running"
]
inputs = package["inputs"]

st.success(
    "Forecast berhasil dibuat "
    "menggunakan saved model."
)

st.caption(
    f"Bulan aktif: "
    f"{inputs['forecast_year']}-"
    f"{inputs['forecast_month']:02d} | "
    f"Cutoff Booking Date: "
    f"{inputs['cutoff_date'].strftime('%Y-%m-%d')} | "
    f"Inventory Hotel: "
    f"{inputs['total_rooms']} kamar"
)


# =========================================================
# DATA CHECK
# =========================================================

st.subheader("Data Check")

data_col_1, data_col_2, data_col_3, data_col_4 = (
    st.columns(4)
)

data_col_1.metric(
    "Total Rows",
    format_number(
        len(df)
    )
)

data_col_2.metric(
    "Total Revenue",
    format_currency(
        df[
            "Current Room Revenue"
        ].sum()
    )
)

data_col_3.metric(
    "Total Rooms Sold",
    format_number(
        df[
            "Current Room Nights"
        ].sum()
    )
)

data_col_4.metric(
    "Latest Stay Date",
    str(
        df[
            "Stay Date"
        ].max().date()
    )
)

st.subheader("Active Month Logic")

logic_col_1, logic_col_2, logic_col_3, logic_col_4 = (
    st.columns(4)
)

logic_col_1.metric(
    "Training Rows",
    format_number(
        len(training_preview)
    )
)

logic_col_2.metric(
    "As-of Rows",
    format_number(
        len(as_of_preview)
    )
)

logic_col_3.metric(
    "Training Last Stay Date",
    (
        str(
            training_preview[
                "Stay Date"
            ].max().date()
        )
        if not training_preview.empty
        else "No data"
    )
)

logic_col_4.metric(
    "Forecast Months",
    format_number(
        inputs[
            "forecast_horizon_months"
        ]
    )
)

with st.expander(
    "Cutoff Date Explanation"
):
    st.markdown(
        """
**Cutoff Date** adalah batas tanggal booking
yang dianggap sudah diketahui oleh sistem.

- **Booking Date sampai Cutoff Date** menentukan
  data yang boleh dipakai.
- **Stay Date** menentukan revenue masuk ke bulan
  forecast yang mana.

Contoh:

- Cutoff Date = 29 Mei
- Forecast Month = Juni

Maka data yang menjadi Known Revenue adalah booking
dengan **Stay Date pada Juni** dan
**Booking Date sampai 29 Mei**.
        """
    )


# =========================================================
# MODEL INFORMATION
# =========================================================

st.subheader("Model Information")

model_col_1, model_col_2, model_col_3, model_col_4 = (
    st.columns(4)
)

model_col_1.metric(
    "Model Source",
    "Saved Model"
)

model_col_2.metric(
    "Linear Regression",
    "Loaded"
)

model_col_3.metric(
    "Naive Bayes",
    "Loaded"
)

model_col_4.metric(
    "Revenue State",
    "Loaded"
)

with st.expander(
    "Loaded Feature Information"
):
    st.write(
        "Linear Regression Features"
    )

    st.write(
        linear_features
    )

    st.write(
        "Naive Bayes Features"
    )

    st.write(
        naive_features
    )


# =========================================================
# SCENARIO AND ADJUSTMENT
# =========================================================

st.subheader(
    "Scenario and Adjustment Control"
)

st.caption(
    "Perubahan demand dan adjustment "
    "baru diterapkan setelah klik "
    "Apply Scenario & Adjustment."
)

if not st.session_state[
    "applied_demand_factor_input"
]:
    st.session_state[
        "applied_demand_factor_input"
    ] = build_default_demand_input(
        forecast_raw
    )

base_summary_for_form = (
    apply_demand_factor(
        forecast_raw,
        st.session_state[
            "applied_demand_factor_input"
        ]
    )
)

if not st.session_state[
    "applied_adjustment_input"
]:
    st.session_state[
        "applied_adjustment_input"
    ] = build_default_adjustment_input(
        base_summary_for_form,
        adjustment_segments
    )

with st.form(
    "scenario_adjustment_form"
):
    st.markdown(
        "### Demand Scenario"
    )

    demand_factor_input_temp = {}

    for _, row in forecast_raw.iterrows():
        period = str(
            row["Forecast Period"]
        )

        period_type = row[
            "Period Type"
        ]

        current_factor = (
            st.session_state[
                "applied_demand_factor_input"
            ].get(
                period,
                1.00
            )
        )

        current_scenario = (
            factor_to_scenario_label(
                current_factor
            )
        )

        default_index = (
            scenario_options.index(
                current_scenario
            )
            if current_scenario
            in scenario_options
            else scenario_options.index(
                "Normal"
            )
        )

        selected_scenario = (
            st.selectbox(
                (
                    f"Demand Scenario - "
                    f"{period} "
                    f"({period_type})"
                ),
                scenario_options,
                index=default_index,
                key=(
                    f"demand_form_"
                    f"{period}"
                )
            )
        )

        demand_factor_input_temp[
            period
        ] = scenario_to_factor(
            selected_scenario
        )

    st.markdown(
        "### Segment Adjustment"
    )

    adjustment_input_temp = {}

    for _, row in (
        base_summary_for_form.iterrows()
    ):
        period = str(
            row["Forecast Period"]
        )

        period_type = row[
            "Period Type"
        ]

        with st.expander(
            (
                f"Adjustment for "
                f"{period_type} - "
                f"{period}"
            ),
            expanded=False
        ):
            adjustment_input_temp[
                period
            ] = {}

            for segment in adjustment_segments:
                current_value = (
                    st.session_state[
                        "applied_adjustment_input"
                    ]
                    .get(
                        period,
                        {}
                    )
                    .get(
                        segment,
                        0
                    )
                )

                adjustment_left, adjustment_right = (
                    st.columns(
                        [2, 1]
                    )
                )

                with adjustment_left:
                    st.write(
                        (
                            f"{segment} | "
                            "Benchmark Average "
                            "Room Rate: "
                            f"{format_currency(rate_benchmark.get(segment, 0))}"
                        )
                    )

                with adjustment_right:
                    adjustment_input_temp[
                        period
                    ][segment] = (
                        st.number_input(
                            (
                                "Additional Rooms "
                                f"Sold - {segment} "
                                f"- {period}"
                            ),
                            min_value=0,
                            value=int(
                                current_value
                            ),
                            step=1,
                            key=(
                                f"adjust_form_"
                                f"{period}_"
                                f"{segment}"
                            )
                        )
                    )

    apply_scenario_button = (
        st.form_submit_button(
            "Apply Scenario & Adjustment"
        )
    )

if apply_scenario_button:
    st.session_state[
        "applied_demand_factor_input"
    ] = demand_factor_input_temp

    st.session_state[
        "applied_adjustment_input"
    ] = adjustment_input_temp

forecast_summary = (
    apply_demand_factor(
        forecast_raw,
        st.session_state[
            "applied_demand_factor_input"
        ]
    )
)

demand_factor_table = pd.DataFrame([
    {
        "Forecast Period":
            period,

        "Demand Factor":
            factor,

        "Demand Scenario":
            factor_to_scenario_label(
                factor
            )
    }
    for period, factor
    in st.session_state[
        "applied_demand_factor_input"
    ].items()
])

adjusted_forecast = (
    apply_segment_adjustment(
        forecast_summary,
        st.session_state[
            "applied_adjustment_input"
        ],
        rate_benchmark
    )
)


# =========================================================
# CURRENT RUNNING PERFORMANCE
# =========================================================

current_month_row = forecast_summary[
    forecast_summary[
        "Period Type"
    ] == "Current Month"
]

if not current_month_row.empty:
    current_month_row = (
        current_month_row.iloc[0]
    )

    current_otb_revenue = (
        current_month_row[
            "Known Revenue"
        ]
    )

    current_otb_rooms_sold = (
        current_month_row[
            "Known Rooms Sold"
        ]
    )

    current_otb_average_room_rate = (
        current_month_row[
            "Known Average Room Rate"
        ]
    )

    current_budget_revenue = (
        current_month_row[
            "Budget Revenue"
        ]
    )

    current_otb_achievement = (
        safe_div(
            current_otb_revenue,
            current_budget_revenue
        ) * 100
    )

    current_revenue_remaining_to_budget = (
        current_budget_revenue -
        current_otb_revenue
    )

else:
    current_otb_revenue = 0
    current_otb_rooms_sold = 0
    current_otb_average_room_rate = 0
    current_budget_revenue = 0
    current_otb_achievement = 0
    current_revenue_remaining_to_budget = 0

st.subheader(
    "Current Running Performance"
)

running_col_1, running_col_2, running_col_3, running_col_4 = (
    st.columns(4)
)

running_col_1.metric(
    "Actual Running Revenue",
    format_currency(
        actual_running[
            "Actual Running Revenue"
        ]
    )
)

running_col_2.metric(
    "Actual Running Rooms Sold",
    format_number(
        actual_running[
            "Actual Running Rooms Sold"
        ]
    )
)

running_col_3.metric(
    "Actual Running Average Room Rate",
    format_currency(
        actual_running[
            "Actual Running Average Room Rate"
        ]
    )
)

running_col_4.metric(
    "OTB Revenue as of Cutoff",
    format_currency(
        current_otb_revenue
    )
)

running_col_5, running_col_6, running_col_7, running_col_8 = (
    st.columns(4)
)

running_col_5.metric(
    "OTB Rooms Sold as of Cutoff",
    format_number(
        current_otb_rooms_sold
    )
)

running_col_6.metric(
    "OTB Revenue Achievement",
    format_percent(
        current_otb_achievement
    )
)

running_col_7.metric(
    "Budget Revenue Current Month",
    format_currency(
        current_budget_revenue
    )
)

running_col_8.metric(
    "Revenue Remaining to Budget",
    format_currency(
        current_revenue_remaining_to_budget
    )
)

with st.expander(
    "Actual Running Revenue vs OTB Revenue"
):
    st.markdown(
        """
**Actual Running Revenue** adalah revenue untuk
Stay Date yang sudah terjadi sampai cutoff.

**OTB Revenue as of Cutoff** adalah seluruh revenue
untuk bulan tersebut yang Booking Date-nya sudah masuk
sampai cutoff, termasuk Stay Date yang masih akan datang.
        """
    )


# =========================================================
# BASELINE FORECAST OVERVIEW
# =========================================================

forecast_summary = (
    add_dominant_revenue_state_to_summary(
        forecast_summary,
        nb_detail
    )
)

st.subheader(
    "Baseline Forecast Overview"
)

cols_per_row = (
    min(
        3,
        len(forecast_summary)
    )
    if len(forecast_summary) > 0
    else 1
)

for start_index in range(
    0,
    len(forecast_summary),
    cols_per_row
):
    columns = st.columns(
        cols_per_row
    )

    chunk = forecast_summary.iloc[
        start_index:
        start_index + cols_per_row
    ]

    for column_index, (
        _,
        row
    ) in enumerate(
        chunk.iterrows()
    ):
        with columns[column_index]:
            st.markdown(
                f"### {row['Period Type']}"
            )

            st.markdown(
                f"**{row['Forecast Period']}**"
            )

            st.metric(
                "Projected Revenue",
                format_currency(
                    row[
                        "Projected Revenue from Naive Bayes"
                    ]
                )
            )

            st.metric(
                "Budget Revenue",
                format_currency(
                    row[
                        "Budget Revenue"
                    ]
                )
            )

            st.metric(
                "Naive Bayes Confidence Score",
                format_percent(
                    row[
                        "Naive Bayes Average Confidence %"
                    ]
                )
            )

            st.metric(
                "Selected Revenue State",
                format_revenue_state_label(
                    row.get(
                        "Dominant Revenue State",
                        "-"
                    )
                )
            )

st.subheader(
    "Forecast Summary Table"
)

st.dataframe(
    format_dataframe_for_display(
        forecast_summary
    ),
    use_container_width=True
)

with st.expander(
    "Known Revenue / OTB Audit",
    expanded=False
):
    st.markdown(
        """
Known Revenue dihitung secara terpisah untuk setiap
bulan forecast menggunakan:

- **Booking Date sampai Cutoff Date**
- **Stay Date pada bulan forecast**

Nilai Current Month, Month + 1, dan Month + 2
seharusnya berbeda sesuai booking yang telah masuk.
        """
    )

    known_audit_cols = [
        "Forecast Period",
        "Period Type",
        "Cutoff Date",
        "Known Booking Rows",
        "Known Revenue",
        "Known Rooms Sold",
        "Known Average Room Rate",
        "Known Future Revenue",
        "Known Future Rooms Sold",
        "Actual Running Revenue",
        "Actual Running Rooms Sold"
    ]

    known_audit_cols = [
        col
        for col in known_audit_cols
        if col in forecast_summary.columns
    ]

    st.dataframe(
        format_dataframe_for_display(
            forecast_summary[
                known_audit_cols
            ]
        ),
        use_container_width=True
    )


# =========================================================
# ADJUSTED FORECAST
# =========================================================

st.subheader(
    "Adjusted Forecast Overview"
)

for start_index in range(
    0,
    len(adjusted_forecast),
    cols_per_row
):
    columns = st.columns(
        cols_per_row
    )

    chunk = adjusted_forecast.iloc[
        start_index:
        start_index + cols_per_row
    ]

    for column_index, (
        _,
        row
    ) in enumerate(
        chunk.iterrows()
    ):
        with columns[column_index]:
            st.markdown(
                f"### {row['Period Type']}"
            )

            st.markdown(
                f"**{row['Forecast Period']}**"
            )

            st.metric(
                "Adjusted Projected Revenue",
                format_currency(
                    row[
                        "Adjusted Projected Revenue"
                    ]
                )
            )

            st.metric(
                "Budget Revenue",
                format_currency(
                    row[
                        "Budget Revenue"
                    ]
                )
            )

            st.metric(
                "Adjusted Revenue Achievement",
                format_percent(
                    row[
                        "Adjusted Revenue Achievement %"
                    ]
                )
            )

            st.metric(
                "Adjustment Rooms Sold",
                format_number(
                    row[
                        "Adjustment Rooms Sold"
                    ]
                )
            )

            st.caption(
                (
                    "Adjusted Rooms Sold: "
                    f"{format_number(row['Adjusted Projected Rooms Sold'])} | "
                    "Adjusted Average Room Rate: "
                    f"{format_currency(row['Adjusted Average Room Rate'])}"
                )
            )

st.subheader(
    "Adjusted Forecast Table"
)

st.dataframe(
    format_dataframe_for_display(
        adjusted_forecast
    ),
    use_container_width=True
)


# =========================================================
# INTERACTIVE VISUALIZATION
# =========================================================

st.subheader(
    "Interactive Visualization"
)

chart_option = st.selectbox(
    "Choose Chart",
    [
        "Revenue vs Budget",
        "Rooms Sold vs Budget",
        "Revenue Achievement"
    ]
)

if chart_option == "Revenue vs Budget":
    st.plotly_chart(
        build_revenue_chart(
            forecast_summary,
            adjusted_forecast
        ),
        use_container_width=True
    )

elif chart_option == "Rooms Sold vs Budget":
    st.plotly_chart(
        build_rooms_chart(
            forecast_summary,
            adjusted_forecast
        ),
        use_container_width=True
    )

else:
    st.plotly_chart(
        build_achievement_chart(
            forecast_summary,
            adjusted_forecast
        ),
        use_container_width=True
    )


# =========================================================
# PICKUP RECOMMENDATION
# =========================================================

st.subheader(
    "Pickup Recommendation: "
    "Group and Transient"
)

pickup_rows = []

for _, row in forecast_summary.iterrows():
    additional_rooms = row[
        "Additional Rooms Sold Needed"
    ]

    allocation = {
        "Group": 0.35,
        "Transient - Corporate/Government":
            0.40,
        "Transient - Retail":
            0.25
    }

    for (
        segment,
        weight
    ) in allocation.items():
        pickup_rows.append({
            "Forecast Period":
                row[
                    "Forecast Period"
                ],

            "Period Type":
                row[
                    "Period Type"
                ],

            "Focus Segment":
                segment,

            "Suggested Additional Rooms Sold":
                round(
                    additional_rooms *
                    weight
                )
        })

pickup_df = pd.DataFrame(
    pickup_rows
)

st.dataframe(
    format_dataframe_for_display(
        pickup_df
    ),
    use_container_width=True
)


# =========================================================
# DETAIL TABLES
# =========================================================

with st.expander(
    "Future Daily Forecast Detail"
):
    st.dataframe(
        format_dataframe_for_display(
            forecast_future
        ),
        use_container_width=True
    )

with st.expander(
    "Naive Bayes Probability Detail"
):
    st.markdown(
        """
Cara bacanya:

- Sistem membaca fitur harian.
- Model menghitung probabilitas Revenue State.
- State dengan probabilitas tertinggi menjadi
  Most Likely Revenue State.
- Expected Revenue dihitung dari probabilitas setiap
  state dikalikan rata-rata revenue state tersebut.

**Expected Revenue =
Probability State × Average Revenue State**
        """
    )

    st.dataframe(
        format_dataframe_for_display(
            nb_detail
        ),
        use_container_width=True
    )

with st.expander(
    "Revenue State Reference"
):
    st.markdown(
        """
Revenue State Reference berasal dari
**bin_reference.pkl**.

- **Revenue State**: label kategori revenue.
- **Minimum Revenue in State**: revenue terendah.
- **Maximum Revenue in State**: revenue tertinggi.
- **Average Revenue in State**: rata-rata revenue yang
  digunakan untuk menghitung expected revenue.
        """
    )

    st.dataframe(
        format_dataframe_for_display(
            bin_reference
        ),
        use_container_width=True
    )

with st.expander(
    "Average Room Rate Cap Detail"
):
    cap_cols = [
        "Forecast Period",
        "Period Type",
        "Known Revenue",
        "Raw Projected Revenue from Naive Bayes before Cap",
        "Raw Average Room Rate before Cap",
        "Average Room Rate Cap",
        "Average Room Rate Cap Applied",
        "Average Room Rate Cap Basis",
        "OTB Floor Overrode Average Room Rate Cap",
        "Projected Revenue from Naive Bayes",
        "Projected Average Room Rate"
    ]

    cap_cols = [
        col
        for col in cap_cols
        if col in forecast_summary.columns
    ]

    st.dataframe(
        format_dataframe_for_display(
            forecast_summary[
                cap_cols
            ]
        ),
        use_container_width=True
    )

with st.expander(
    "Adjustment Rate Benchmark"
):
    st.markdown(
        """
Adjustment Revenue dihitung dengan:

**Additional Rooms Sold ×
Benchmark Average Room Rate Segment**
        """
    )

    st.dataframe(
        format_dataframe_for_display(
            rate_benchmark_table
        ),
        use_container_width=True
    )

with st.expander(
    "Budget Master"
):
    st.dataframe(
        format_dataframe_for_display(
            budget_df
        ),
        use_container_width=True
    )


# =========================================================
# DOWNLOAD OUTPUT
# =========================================================

output = BytesIO()

with pd.ExcelWriter(
    output,
    engine="openpyxl"
) as writer:
    forecast_summary.to_excel(
        writer,
        sheet_name="Baseline Forecast",
        index=False
    )

    adjusted_forecast.to_excel(
        writer,
        sheet_name="Adjusted Forecast",
        index=False
    )

    demand_factor_table.to_excel(
        writer,
        sheet_name="Demand Factor",
        index=False
    )

    pickup_df.to_excel(
        writer,
        sheet_name="Pickup Recommendation",
        index=False
    )

    forecast_future.to_excel(
        writer,
        sheet_name="Future Daily Forecast",
        index=False
    )

    nb_detail.to_excel(
        writer,
        sheet_name="Naive Bayes Detail",
        index=False
    )

    bin_reference.to_excel(
        writer,
        sheet_name="Revenue State",
        index=False
    )

    rate_benchmark_table.to_excel(
        writer,
        sheet_name="Rate Benchmark",
        index=False
    )

    budget_df.to_excel(
        writer,
        sheet_name="Budget Master",
        index=False
    )

    known_audit_cols = [
        "Forecast Period",
        "Period Type",
        "Cutoff Date",
        "Known Booking Rows",
        "Known Revenue",
        "Known Rooms Sold",
        "Known Average Room Rate",
        "Known Future Revenue",
        "Known Future Rooms Sold",
        "Actual Running Revenue",
        "Actual Running Rooms Sold",
        "Actual Running Average Room Rate"
    ]

    known_audit_cols = [
        col
        for col in known_audit_cols
        if col in forecast_summary.columns
    ]

    forecast_summary[
        known_audit_cols
    ].to_excel(
        writer,
        sheet_name="Known Revenue Audit",
        index=False
    )

    cap_cols = [
        "Forecast Period",
        "Period Type",
        "Known Revenue",
        "Raw Projected Revenue from Naive Bayes before Cap",
        "Raw Average Room Rate before Cap",
        "Average Room Rate Cap",
        "Average Room Rate Cap Applied",
        "Average Room Rate Cap Basis",
        "OTB Floor Overrode Average Room Rate Cap",
        "Projected Revenue from Naive Bayes",
        "Projected Average Room Rate"
    ]

    cap_cols = [
        col
        for col in cap_cols
        if col in forecast_summary.columns
    ]

    forecast_summary[
        cap_cols
    ].to_excel(
        writer,
        sheet_name="Average Rate Cap",
        index=False
    )

st.download_button(
    label=(
        "Download Forecast Output Excel"
    ),
    data=output.getvalue(),
    file_name=(
        "hotel_forecast_output_fixed.xlsx"
    ),
    mime=(
        "application/vnd.openxmlformats-"
        "officedocument.spreadsheetml.sheet"
    )
)