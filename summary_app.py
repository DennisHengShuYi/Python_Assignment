import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import altair as alt

st.set_page_config(page_title="Memecoin Mania Summary", layout="wide")

st.title("📈 Memecoin Mania: Summary & Insights")

# --- Task 1: Intro Section ---
st.header("🪙 Task 1: What Are Memecoins?")

st.write("""
Memecoins are altcoins based on internet memes or viral trends. Most lack real utility and are driven by hype, humor, and speculation rather than technology or value.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📌 Key Facts")
    st.markdown("""
    - Fun coins with little to no use case  
    - Named after memes/pop culture  
    - Backed by hype, not assets  
    - Mostly for speculation  
    """)

with col2:
    st.subheader("🚀 Why Popular?")
    st.markdown("""
    - Social media buzz  
    - Celebrity tweets (e.g., Elon Musk)  
    - Strong communities  
    - Volatile = quick gains  
    - Cheap to buy  
    """)

with col3:
    st.subheader("🌟 Examples")
    st.markdown("""
    - **DOGE** – Joke coin turned viral  
    - **SHIB** – Inspired by DOGE  
    - **PEPE** – Meme frog coin  
    """)
    st.caption("📈 As of Feb 20, 2025, DOGE, SHIB, and PEPE made up 67% of the meme coin market.")


# --- Task 2: Data Collection & Preprocessing ---
st.header("📊 Task 2: Data Collection & Preprocessing")

with st.expander("📍 Ripple Market Data"):
    st.markdown("""
    - **Source**: CoinGecko API  
    - **Duration**: May 2024 – June 2025 (366 days)  
    - **Attributes**: Price, Market Cap, Volume, Rank, Supply, All-Time Highs  
    - **Stored in**: `ripple_market_data.csv`
    """)
    try:
        ripple_df = pd.read_csv("ripple_market_data.csv")
        ripple_df["date"] = pd.to_datetime(ripple_df["Timestamp"]).dt.date
        st.dataframe(ripple_df.head())
        st.line_chart(ripple_df.set_index("date")["Price_USD"])
    except FileNotFoundError:
        st.warning("⚠️ ripple_market_data.csv not found.")

with st.expander("💹 All Memecoins Market Data"):
    st.markdown("""
    - **Coins**: Bitcoin, Shiba Inu, Dogecoin, Cardano  
    - **Duration**: 366 records per coin (~1464 total)  
    - **Stored in**: `all_memecoins_market_data.csv`  
    - Includes additional columns: Daily Return %, 7-day rolling mean/std, Market Cap per Coin
    """)
    try:
        memecoins_df = pd.read_csv("all_memecoins_market_data.csv")
        memecoins_df["date"] = pd.to_datetime(memecoins_df["Timestamp"]).dt.date
        selected = st.selectbox("Select a Coin", memecoins_df["Symbol"].unique())
        filtered = memecoins_df[memecoins_df["Symbol"] == selected]
        st.dataframe(filtered.head())
        st.line_chart(filtered.set_index("date")["Price_USD"])
    except FileNotFoundError:
        st.warning("⚠️ all_memecoins_market_data.csv not found.")

with st.expander("👾 Reddit Posts (Ripple)"):
    st.markdown("""
    - **Data**: Ripple-related Reddit posts & comments  
    - **Tool**: `praw` Reddit API  
    - **Records**: 318 posts (~100/month over 4 months)  
    - **Stored in**: `xrp_reddit_data.csv`
    """)
    try:
        reddit_df = pd.read_csv("xrp_reddit_data.csv")
        reddit_df["full_text"] = reddit_df["post_content"].astype(str) + " " + reddit_df["comments"].astype(str)
        st.dataframe(reddit_df[["created_at", "post_upvotes", "num_comments", "full_text"]].head())
    except FileNotFoundError:
        st.warning("⚠️ xrp_reddit_data.csv not found.")

with st.expander("📰 News Headlines"):
    st.markdown("""
    - **Articles**: 100 memecoin-related news articles  
    - **Duration**: 1 year  
    - **Attributes**: Title, description, URL, source, published date  
    - **Stored in**: `memecoin_news.csv`  
    - **API Used**: NewsAPI (key: `6a913b6396024d10a3556536527e18e1`)
    """)
    try:
        news_df = pd.read_csv("memecoin_news.csv")
        st.dataframe(news_df[["published_at", "title", "source"]].head())
    except FileNotFoundError:
        st.warning("⚠️ memecoin_news.csv not found.")

with st.expander("🔍 Google Search Trends"):
    st.markdown("""
    - **Search terms**: dogecoin, shiba inu, bitcoin, cardano  
    - **Records**: 100 weekly entries  
    - **Duration**: 1 year (Google Trends caps at 1-year window)  
    - **Stored in**: `search_trends_extended.csv`
    """)
    try:
        trends_df = pd.read_csv("search_trends_extended.csv")
        trends_df["date"] = pd.to_datetime(trends_df["date"])
        st.line_chart(trends_df.set_index("date")[["dogecoin", "shiba inu", "bitcoin", "cardano"]])
        st.dataframe(trends_df.head())
    except FileNotFoundError:
        st.warning("⚠️ search_trends_extended.csv not found.")


# --- Task 3: Sentiment Analysis ---
st.header("📈 Task 3: Sentiment Analysis - Dogecoin vs XRP")

# Load data
@st.cache_data
def load_data():
    doge_df = pd.read_csv("doge_adjusted_market_sentiment_search.csv", parse_dates=["date"])
    xrp_df = pd.read_csv("xrp_sentiment_dashboard_ready.csv", parse_dates=["date"])
    return doge_df, xrp_df

doge_df, xrp_df = load_data()

# Layout columns
col1, col2 = st.columns(2)

# --- DOGE SECTION ---
with col1:
    st.subheader("🐶 Dogecoin")

    # Filter to your available date range
    doge_filtered = doge_df[
        (doge_df['date'] >= pd.to_datetime("2025-04-21")) &
        (doge_df['date'] < pd.to_datetime("2025-05-29"))
    ]

    # Line chart inside expander
    with st.expander("📈 Select Dogecoin Line Chart View"):
        doge_option = st.radio(
            "Select daily metric to display over time (Dogecoin):",
            ["Reddit Sentiment", "Search Trend"]
        )

        doge_daily = doge_filtered.groupby('date').agg({
            'reddit_sentiment': 'mean',
            'search_trend': 'mean'
        }).reset_index()

        if doge_option == "Reddit Sentiment":
            st.line_chart(
                doge_daily.rename(columns={"date": "Date", "reddit_sentiment": "Value"})
                          .set_index("Date")["Value"]
            )
        else:
            st.line_chart(
                doge_daily.rename(columns={"date": "Date", "search_trend": "Value"})
                          .set_index("Date")["Value"]
            )

    # Distribution chart inside expander
    with st.expander("📊 Dogecoin Distribution View"):
        doge_dist_option = st.radio(
            "Select distribution metric (Dogecoin):",
            ["Reddit Sentiment", "Search Trend"]
        )

        fig, ax = plt.subplots()
        if doge_dist_option == "Reddit Sentiment":
            sns.histplot(
                doge_filtered['reddit_sentiment'].dropna(),
                bins=20, kde=True, color="orange", ax=ax
            )
            ax.set_title("Reddit Sentiment Distribution")
        else:
            sns.histplot(
                doge_filtered['search_trend'].dropna(),
                bins=20, kde=True, color="green", ax=ax
            )
            ax.set_title("Search Trend Distribution")

        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)


# --- XRP SECTION ---
with col2:
    st.subheader("💠 XRP ")

    # Line chart with expander
    with st.expander("📈 Select XRP Line Chart View"):
        xrp_option = st.radio("Choose metric to display over time (XRP):", ["Reddit Sentiment", "Search Trend"])

        xrp_daily = xrp_df.groupby(xrp_df['date'].dt.date).agg({
            "reddit_sentiment": "mean",
            "search_trend": "mean"
        }).reset_index()

        if xrp_option == "Reddit Sentiment":
            st.line_chart(xrp_daily.rename(columns={"date": "Date", "reddit_sentiment": "Value"}).set_index("Date")["Value"])
        else:
            st.line_chart(xrp_daily.rename(columns={"date": "Date", "search_trend": "Value"}).set_index("Date")["Value"])

    # Distribution chart with expander
    with st.expander("📊 XRP Distribution View"):
        xrp_dist_option = st.radio("Choose distribution to display (XRP):", ["Reddit Sentiment", "Search Trend"])

        fig2, ax2 = plt.subplots()
        if xrp_dist_option == "Reddit Sentiment":
            sns.histplot(xrp_df['reddit_sentiment'].dropna(), bins=30, kde=True, ax=ax2, color="blue")
            ax2.set_title("Distribution of Reddit Sentiment (XRP)")
        else:
            sns.histplot(xrp_df['search_trend'].dropna(), bins=30, kde=True, ax=ax2, color="purple")
            ax2.set_title("Distribution of Search Trend (XRP)")
        ax2.set_xlabel("Score")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)

# Show data previews (optional)
with st.expander("🔍 View Raw Data"):
    st.write("Dogecoin Data Sample:")
    st.dataframe(doge_df.head())
    st.write("XRP Data Sample:")
    st.dataframe(xrp_df.head())


# --- Task 4: Trend Analysis ---
st.header("📈 Task 4: Trend Analysis - Dogecoin vs XRP")

# Feature categories and examples
feature_summary = {
    "Feature Category": [
        "Lag Features",
        "Technical Indicators",
        "Returns/Volatility",
        "Sentiment Features",
        "Search Trend Features",
        "Date/Time Features"
    ],
    "Examples": [
        "Price_USD_lag1, reddit_sentiment_lag1",
        "Price_USD_ema_7, RSI_14, MACD",
        "Price_USD_return_1d, volatility_7",
        "reddit_sentiment_roll3, momentum_1",
        "search_trend_lag1, roll3, volatility_7",
        "day_of_week, is_weekend, month, quarter"
    ],
    "Purpose": [
        "Capture price/sentiment momentum",
        "Identify technical trading patterns",
        "Measure fluctuation and daily change",
        "Track Reddit sentiment shifts",
        "Model public interest dynamics",
        "Introduce temporal patterns"
    ]
}

# Show Feature Engineering Summary
with st.expander("📐 Feature Engineering Summary"):
    st.dataframe(pd.DataFrame(feature_summary))

# Layout columns
col1, col2 = st.columns(2)

# --- DOGE SECTION ---
with col1:
    st.subheader("🐶 Dogecoin")

    # Distribution chart inside expander
    with st.expander("📊 Dogecoin Trend Analysis"):
        doge_dist_option = st.radio(
            "Select distribution metric (Dogecoin):",
            ["Price Over Time", "Trading Volume Over Time"]
        )

        # Convert 'Timestamp' to datetime and set it as the index (only once)
        if doge_df.index.name != "Timestamp":
            doge_df["Timestamp"] = pd.to_datetime(doge_df["Timestamp"])
            doge_df.set_index("Timestamp", inplace=True)

        if doge_dist_option == "Price Over Time":
            st.line_chart(doge_df["Price_USD"])

        else:
            st.line_chart(doge_df["Total_Volume_USD"])

        # Summary Stats + Data Inspection
        st.subheader("🧾 Data Inspection: Structure & Quality")

        # Display column names, data types, and non-null counts in a DataFrame
        data_info = pd.DataFrame({
            "Column": doge_df.columns,
            "Data Type": doge_df.dtypes.values,
            "Non-Null Count": doge_df.notnull().sum().values
        })

        st.dataframe(data_info)

        # Missing values
        st.markdown("❓ **Missing Values per Column:**")

        missing_df = doge_df.isnull().sum().reset_index()
        missing_df.columns = ["Column", "Missing Values"]
        missing_df = missing_df[missing_df["Missing Values"] > 0]

        if missing_df.empty:
            st.success("✅ No missing values detected in the dataset.")
        else:
            st.dataframe(missing_df.sort_values(by="Missing Values", ascending=False))

    with st.expander("📈 Dogecoin Price vs Sentiment Score Over Time"):

        # Ensure Timestamp is datetime and set as index once
        if doge_df.index.name != "Timestamp":
            doge_df["Timestamp"] = pd.to_datetime(doge_df["Timestamp"])
            doge_df.set_index("Timestamp", inplace=True)

        # Filter for the desired date range
        start_date = pd.to_datetime("2025-04-21")
        end_date = pd.to_datetime("2025-05-29")
        df_doge_filtered = doge_df.loc[(doge_df.index >= start_date) & (doge_df.index <= end_date)]

        # Build Plotly figure
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_doge_filtered.index,
            y=df_doge_filtered["Price_USD"],
            name="Dogecoin Price (USD)",
            line=dict(color="blue", width=2),
            yaxis="y1"
        ))

        fig.add_trace(go.Scatter(
            x=df_doge_filtered.index,
            y=df_doge_filtered["reddit_sentiment"],
            name="Sentiment Score",
            line=dict(color="orange", width=1.5),
            yaxis="y2"
        ))

        fig.update_layout(
            title="Dogecoin Price vs Sentiment Score Over Time",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Price (USD)", side="left", showgrid=False),
            yaxis2=dict(
                title="Sentiment Score",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            legend=dict(x=0.01, y=0.99),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)


    # Show Feature Engineering Summary
    with st.expander("📐 Correlation Matrix of Dogecoin's Features"):

        # Step 1: Identify original (non-engineered) features
        doge_feature_cols = [
            "reddit_sentiment",
            "Price_USD_rsi_14",
            "Price_USD_volatility_7",
            "Price_USD_return_1d",
            "day_of_week", "is_weekend",
            "Price_USD_macd", "Price_USD_macd_signal",
            "reddit_sentiment_volatility_7", "reddit_sentiment_momentum_1",
            "month", "quarter",
            "search_trend", "search_trend_momentum_1", "search_trend_volatility_7"
        ]
        target_col = "Price_USD"

        # Step 2: Compute correlation with target
        all_cols_for_corr = doge_feature_cols + [target_col]
        existing_cols = [col for col in all_cols_for_corr if col in xrp_df.columns]

        correlations = xrp_df[existing_cols].corr()

        # --- New Section: Displaying the complete correlation list ---
        st.subheader("📊 Complete Correlation Matrix with Target")
        st.write(f"This table shows how each feature correlates with `{target_col}`, sorted by the strength of the correlation (both positive and negative).")

        # Extract correlation with target, sort by absolute value, and display
        full_corr_with_target = correlations[target_col].drop(target_col).sort_values(key=abs, ascending=False)
        st.dataframe(full_corr_with_target.reset_index().rename(columns={"index": "Feature", target_col: "Correlation Coefficient"}), use_container_width=True)


        # --- Section for Top 5 Features ---
        st.subheader("🎯 Top 5 Most Correlated Features")

        # Step 3: Extract top 5 correlated features (positive or negative)
        top_corr = correlations[target_col].drop(target_col).abs().sort_values(ascending=False).head(5)
        top_features = top_corr.index.tolist()

        # Show the top 5 values in a table
        st.write(f"These are the 5 features from your list most strongly correlated with `{target_col}`:")
        top_corr_values = correlations[target_col][top_features].sort_values(key=abs, ascending=False)
        st.table(top_corr_values.reset_index().rename(columns={"index": "Feature", target_col: "Correlation Coefficient"}))

        # Step 4: Create and display a heatmap for these top 5 features
        st.write("### Correlation Heatmap for Top 5 Features")
        fig, ax = plt.subplots(figsize=(10, 7))

        # Create the heatmap using the correlation of only the top 5 features + target
        sns.heatmap(
            xrp_df[top_features + [target_col]].corr(),
            annot=True,       # Show the correlation values on the map
            cmap="coolwarm",  # Use a color map that shows positive (hot) and negative (cold) correlations
            fmt=".2f",        # Format numbers to two decimal places
            linewidths=.5,    # Add lines between cells
            ax=ax
        )

        ax.set_title(f'Heatmap of Top 5 Correlated Features and {target_col}', fontsize=14)
        st.pyplot(fig)



# --- XRP SECTION ---
with col2:
    st.subheader("💠 XRP ")

    # XRP Trend Analysis inside expander
    with st.expander("📊 XRP Trend Analysis"):
        xrp_dist_option = st.radio(
            "Select distribution metric (XRP):",
            ["Price Over Time", "Trading Volume Over Time"]
        )

        # Convert 'Timestamp' to datetime and set it as the index
        xrp_df['Timestamp'] = pd.to_datetime(xrp_df['Timestamp']).dt.date
        xrp_df = xrp_df.set_index("Timestamp")

        if xrp_dist_option == "Price Over Time":
            st.line_chart(xrp_df["Price_USD"])

        else:
            st.line_chart(xrp_df["Total_Volume_USD"])

        # Summary Stats + Data Inspection
        st.subheader("🧾 Data Inspection: Structure & Quality")
   
        # Display column names, data types, and non-null counts in a DataFrame
        data_info = pd.DataFrame({
            "Column": xrp_df.columns,
            "Data Type": xrp_df.dtypes.values,
            "Non-Null Count": xrp_df.notnull().sum().values
        })

        st.dataframe(data_info)

        # Missing values
        st.markdown("❓ **Missing Values per Column:**")

        missing_df = xrp_df.isnull().sum().reset_index()
        missing_df.columns = ["Column", "Missing Values"]
        missing_df = missing_df[missing_df["Missing Values"] > 0]

        if missing_df.empty:
            st.success("✅ No missing values detected in the dataset.")
        else:
            st.dataframe(missing_df.sort_values(by="Missing Values", ascending=False))

    with st.expander("XRP Price vs Sentiment Score Over Time"):
        # Ensure Timestamp is datetime and set as index once
        if xrp_df.index.name != "Timestamp":
            xrp_df["Timestamp"] = pd.to_datetime(xrp_df["Timestamp"])
            xrp_df.set_index("Timestamp", inplace=True)

        # Plotly Figure
        fig = go.Figure()

        # XRP Price line - strong solid color
        fig.add_trace(go.Scatter(
            x=xrp_df.index,
            y=xrp_df["Price_USD"],
            name="XRP Price (USD)",
            line=dict(color="teal", width=2)
        ))

        # Sentiment line - lighter and thinner
        fig.add_trace(go.Scatter(
            x=xrp_df.index,
            y=xrp_df["reddit_sentiment"],
            name="Reddit Sentiment Score",
            yaxis="y2",
            line=dict(color="orange", width=1.5, dash='dot'),
            opacity=0.7
        ))

        fig.update_layout(
            title="XRP Price vs Reddit Sentiment Over Time",
            xaxis=dict(title="Date"),
            yaxis=dict(title="XRP Price (USD)", side="left"),
            yaxis2=dict(title="Sentiment Score", overlaying="y", side="right"),
            legend=dict(x=0.01, y=0.99),
            margin=dict(l=40, r=40, t=60, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)

# --- Task 5: Predictive Modelling ---
st.header("🚀 Task 5: Predictive Modelling")

with st.expander("🐶 Dogecoin Price Prediction - Model Performance"):

    # Tab layout for Modeling
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗂️ A. Modeling Overview",
        "📈 B. Feature & Target Selection",
        "🧠 C. Model Results",
        "🧾 D. Feature Importance",
        "📊 E. Model Graph: Predicted vs Actual"
    ])

    with tab1:
        st.markdown("""
        **Objective**: Predict the next-day closing price of **Dogecoin (USD)**.

        **Features used include:**
        - **Lagged prices**: Previous closing prices (`Price_USD_lag1`, `Price_USD_lag2`, `Price_USD_lag3`)
        - **Reddit sentiment**: Current, lagged, rolling sentiment, volatility, and momentum
        - **Technical indicators**: 
            - Moving averages: EMA (`Price_USD_ema_7`, `Price_USD_ema_14`), SMA (`Price_USD_sma_7`, `Price_USD_sma_14`)
            - Momentum indicators: RSI (`Price_USD_rsi_14`), MACD and signal line
            - Volatility indicators: Bollinger Bands (`bb_upper`, `bb_lower`), price volatility
            - Returns: 1-day and 3-day percentage change
        - **Time-related features**: `day_of_week`, `is_weekend`, `month`, `quarter`
        - **Google search trends**: Current, lagged, and rolling averages (`search_trend`, etc.)

        **Modeling Techniques:**
        - Gradient Boosting
        - Random Forest
        - XGBoost
        - Long Short-Term Memory (LSTM) Neural Network
        """)

    with tab2:
        st.markdown("**Target Variable:** `Price_USD`")
        sample_data = doge_df.head()
        st.dataframe(sample_data, use_container_width=True)

        split_idx = int(len(doge_df) * 0.8)
        train_size = split_idx
        test_size = len(doge_df) - split_idx
        st.markdown("### 🔍 Train-Test Split")

        st.markdown(f"""
        - Chronological split (no shuffling) ensures that future data is not leaked into the past — important for time series prediction.
        - We use an **80/20 split**, where 80% of the data is used for training and 20% for testing.
        - **Training size**: {train_size} rows  
        - **Testing size**: {test_size} rows
        """)

    with tab3:
        st.markdown("### 🔍 Model Comparison & Metrics")

        results = pd.DataFrame({
            "Model": ["Gradient Boosting", "Random Forest", "XGBoost", "LSTM"],
            "MSE": [0.000042, 0.000087, 0.000088, 0.001287],
            "MAE": [0.004899, 0.007363, 0.006788, 0.026689],
            "R²": [0.948458, 0.893512, 0.891539, 0.779171]
        })

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🔶 MSE")
            st.info("📉 MSE (Mean Squared Error): Penalizes large errors more. Lower is better.")
            fig_mse = px.bar(results, x="Model", y="MSE", color_discrete_sequence=["#FFA07A"])
            st.plotly_chart(fig_mse, use_container_width=True)

        with col2:
            st.markdown("### 🔷 MAE")
            st.info("📉 MAE (Mean Absolute Error): Average magnitude of errors. Lower is better.")
            fig_mae = px.bar(results, x="Model", y="MAE", color_discrete_sequence=["#87CEFA"])
            st.plotly_chart(fig_mae, use_container_width=True)

        with col3:
            st.markdown("### 🔸 R²")
            st.info("📈 R² (R-squared): Percentage of variance explained by the model. Closer to 1 is better.")
            fig_r2 = px.bar(results, x="Model", y="R²", color_discrete_sequence=["#90EE90"])
            st.plotly_chart(fig_r2, use_container_width=True)

        st.markdown("### 🧠 Best Parameters from Grid Search")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            **✅ Gradient Boosting**  
            ```python
            {
                'learning_rate': 0.01,
                'max_depth': 3,
                'min_samples_split': 2,
                'n_estimators': 200
            }
            ```
                                """)

        with col2:
            st.markdown("""
            **✅ Random Forest**  
            ```python
            {
                'max_depth': 10,
                'max_features': 'sqrt',
                'min_samples_leaf': 1,
                'n_estimators': 300
            }
            ```
                                """)

        with col3:
            st.markdown("""
            **✅ XGBoost**  
            ```python
            {
                'learning_rate': 0.05,
                'max_depth': 4,
                'n_estimators': 150,
                'subsample': 0.8
            }
            ```
                                """)
    
        st.markdown("""
        ❌ **LSTM**: Tuned manually with Keras, not GridSearchCV.
        """)

       
    with tab4:
        st.markdown("### ⭐ Random Forest: Top 10 Feature Importances")

        top_features = pd.DataFrame({
            "feature": [
                "Price_USD_lag1", "Price_USD_ema_7", "Price_USD_sma_7", "Price_USD_lag2", 
                "Price_USD_sma_14", "Price_USD_lag3", "Price_USD_macd", "Price_USD_ema_14", 
                "bb_upper", "Price_USD_rsi_14"
            ],
            "importance": [
                0.471552, 0.346082, 0.074213, 0.035719, 0.018368, 
                0.017217, 0.015887, 0.010593, 0.002686, 0.002093
            ]
        })

        st.dataframe(top_features, use_container_width=True)

    with tab5:
        st.subheader("📈 Model Prediction vs Actual Comparison")

        # Let user select the model
        model_choice = st.selectbox(
            "Select a model to view its predictions:",
            ("Random Forest", "XGBoost", "Gradient Boosting", "LSTM")
        )

        # Load predictions for all models (assumes single CSV file)
        pred_df = pd.read_csv("doge_model_predictions.csv")

        # Get the predicted column based on user choice
        selected_pred = pred_df[model_choice]

        # Determine if the selected model is the best one
        is_best = model_choice == "Gradient Boosting"

        # Optional subtitle
        title_text = f"{model_choice} - Predicted vs Actual"
        if is_best:
            title_text += " ⭐ (Best Model)"

        # Create interactive scatter plot
        fig = px.scatter(
            x=pred_df["Actual"],
            y=selected_pred,
            labels={"x": "Actual", "y": "Predicted"},
            opacity=0.7,
            trendline="ols"
        )

        # Style the scatter and trendline
        fig.update_traces(marker=dict(color='cyan'), selector=dict(mode='markers'))
        fig.update_traces(line=dict(color='red', dash='dash'), selector=dict(mode='lines'))
        fig.update_layout(title=title_text)

        st.plotly_chart(fig, use_container_width=True)


with st.expander("💠 XRP Price Prediction - Model Performance"):

    # Tab layout for Modeling
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗂️ A. Modeling Overview",
        "📈 B. Feature & Target Selection",
        "🧠 C. Model Results",
        "📊 D. Model Graph: Predicted vs Actual"
    ])

    with tab1:
        st.markdown("""
        **Objective**: Predict the next-day closing price of **XRP (USD)** using historical price, sentiment, and technical data.

        **Features used include:**
        - Lagged prices (e.g., `Price_USD_lag1`, `Price_USD_lag2`, etc.)
        - Reddit sentiment scores and trends
        - Technical indicators (EMA, RSI, SMA, MACD, Bollinger Bands)
        - Day-of-week and calendar features (e.g., `day_of_week`, `month`)
        - Google search interest and its lag/momentum

        **Modeling Techniques Applied**:
        - Support Vector Regression (SVR)
        - XGBoost Regressor
        - Random Forest Regressor
        - LSTM (deep learning)
        """)


    with tab2:

        st.markdown("**Target Variable:** `Price_USD`")
        sample_data = xrp_df.head()
        st.dataframe(sample_data, use_container_width=True)

        split_idx = int(len(xrp_df) * 0.8)
        train_size = split_idx
        test_size = len(xrp_df) - split_idx
        st.markdown("### 🔍 Train-Test Split")

        st.markdown(f"""
        - Chronological split (no shuffling) ensures that future data is not leaked into the past — important for time series prediction.
        - We use an **80/20 split**, where 80% of the data is used for training and 20% for testing.
        - **Training size**: {train_size} rows  
        - **Testing size**: {test_size} rows
        """)

    with tab3:
        st.markdown("### 🔍 Model Comparison & Metrics")

        results = pd.DataFrame({
            "Model": ["SVR", "XGBoost", "Random Forest", "LSTM"],
            "RMSE": [0.011219, 0.071677, 0.090603, 0.132339],
            "MAE": [0.009216, 0.056527, 0.065075, 0.104164],
            "R²": [0.994419, 0.772218, 0.636040, 0.226473]
        })

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🔶 MSE")
            st.info("📉 RMSE (Root Mean Squared Error): Penalizes large errors more. Lower is better.")
            fig_mse = px.bar(results, x="Model", y="RMSE", color_discrete_sequence=["#FFA07A"])
            st.plotly_chart(fig_mse, use_container_width=True)

        with col2:
            st.markdown("### 🔷 MAE")
            st.info("📉 MAE (Mean Absolute Error): Average magnitude of errors. Lower is better.")
            fig_mae = px.bar(results, x="Model", y="MAE", color_discrete_sequence=["#87CEFA"])
            st.plotly_chart(fig_mae, use_container_width=True)

        with col3:
            st.markdown("### 🔸 R²")
            st.info("📈 R² (R-squared): Percentage of variance explained by the model. Closer to 1 is better.")
            fig_r2 = px.bar(results, x="Model", y="R²", color_discrete_sequence=["#90EE90"])
            st.plotly_chart(fig_r2, use_container_width=True)

        st.markdown("### 🧠 Best Parameters from Grid Search")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            **✅ XGBoost**  
            ```python
            {
                'colsample_bytree': 0.7,
                'learning_rate': 0.1,
                'max_depth': 3,
                'n_estimators': 150,
                'subsample': 0.7
            }
            ```
            """)

        with col2:
            st.markdown("""
            **✅ Random Forest**  
            ```python
            {
                'max_depth': None,
                'n_estimators': 100
            }
            ```
            """)

        with col3:
            st.markdown("""
            **✅ XGBoost**  
            ```python
            {
                'svr__C': 100,
                'svr__epsilon': 0.01,
                'svr__gamma': 0.001,
                'svr__kernel': 'rbf'
            }
            ```
            """)
    
        st.markdown("""
        ❌ **LSTM**: Tuned manually with Keras, not GridSearchCV.
        """)


    with tab4:
        st.subheader("📊 Model Comparison: Predicted vs Actual (XRP)")

        # Load all predictions
        pred_df = pd.read_csv("xrp_model_predictions.csv")

        # Let user select a model
        selected_model = st.selectbox(
            "Choose a model to view predictions:",
            ["Random Forest", "XGBoost", "SVR", "LSTM"]
        )

        # Set the best model name manually
        best_model = "SVR"  # You can dynamically choose this from a table if needed

        # Determine actual column based on model
        actual_col = "LSTM_Actual" if selected_model == "LSTM" else "Actual"

        # Display title with note if best model
        model_title = f"{selected_model} - Predicted vs Actual"
        if selected_model == best_model:
            model_title += " (Best Model)"

        # Layout in two columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📌 Scatter Plot")

            # Create scatter plot
            scatter_fig = px.scatter(
                pred_df,
                x=actual_col,
                y=selected_model,
                opacity=0.7,
                trendline="ols",
                labels={actual_col: "Actual", selected_model: "Predicted"}
            )

            scatter_fig.update_traces(marker=dict(color='cyan'), selector=dict(mode='markers'))
            scatter_fig.update_traces(line=dict(color='red', dash='dash'), selector=dict(mode='lines'))
            scatter_fig.update_layout(title=model_title)
            st.plotly_chart(scatter_fig, use_container_width=True)

        with col2:
            st.subheader("📈 Line Plot (Over Time)")

            # Check for Date/Time column
            if "Date" in pred_df.columns:
                x_col = "Date"
            elif "Timestamp" in pred_df.columns:
                x_col = "Timestamp"
            else:
                st.warning("No date or timestamp column found for line plot.")
                x_col = None

            if x_col:
                line_fig = px.line(
                    pred_df,
                    x=x_col,
                    y=[actual_col, selected_model],
                    markers=True,
                    labels={"value": "Price", "variable": "Legend"},
                    title="Price Comparison Over Time"
                )
                line_fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)")
                st.plotly_chart(line_fig, use_container_width=True)


st.header(" Task 6: Comparative Analysis")

with st.expander("Comparative analysis: Dogecoin vs XRP"):

    tab1, tab2, tab3 = st.tabs([
        "Market Behavior", "Sentiment Analysis", "Model Comparison"
    ])

    with tab1:
        st.subheader("📈 Market Behavior Comparison: XRP vs DOGE")

        # --- Two-column layout for volatility line chart + distribution chart ---
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Volatility Over Time (7-Day Rolling)")

            # Calculate 7-day rolling volatility
            volatility_xrp = xrp_df[["date", "Price_USD"]].copy()
            volatility_doge = doge_df[["date", "Price_USD"]].copy()

            volatility_xrp["volatility"] = volatility_xrp["Price_USD"].rolling(7).std()
            volatility_doge["volatility"] = volatility_doge["Price_USD"].rolling(7).std()

            volatility_xrp["Asset"] = "XRP"
            volatility_doge["Asset"] = "DOGE"

            combined_volatility = pd.concat([
                volatility_xrp[["date", "volatility", "Asset"]],
                volatility_doge[["date", "volatility", "Asset"]]
            ])

            fig_volatility = px.line(
                combined_volatility,
                x="date",
                y="volatility",
                color="Asset",
                title="📈 7-Day Rolling Price Volatility: XRP vs DOGE",
                labels={"volatility": "Volatility (USD)", "date": "Date"},
                color_discrete_map={"XRP": "#1f77b4", "DOGE": "#FFD700"}
            )

            st.plotly_chart(fig_volatility, use_container_width=True)

        with col2:
            st.markdown("### 🔍 Distribution Comparison by Feature")

            volatility_features = [
                "Price_USD_volatility_7",
                "Price_USD_rsi_14",
                "Price_USD_return_1d"
            ]
            selected_feature = st.selectbox("Select a feature:", volatility_features)

            if selected_feature  in xrp_df.columns and selected_feature  in doge_df.columns:
                xrp_data = xrp_df[selected_feature ].dropna()
                doge_data = doge_df[selected_feature ].dropna()

                if len(xrp_data) > 0 and len(doge_data) > 0:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.set_title(f"Distribution Comparison for {selected_feature }", fontsize=14, fontweight='bold')

                    sns.histplot(xrp_data, kde=True, color="skyblue", label="XRP", stat="density", element="step", ax=ax)
                    sns.histplot(doge_data, kde=True, color="orange", label="DOGE", stat="density", element="step", ax=ax)

                    ax.legend()
                    ax.set_xlabel(selected_feature )
                    ax.set_ylabel("Density")
                    st.pyplot(fig)  # Streamlit compatible

        # --- Section 2: Statistical Comparison Table ---
        st.markdown("### 📊 Comparison Test Results")

        stats_data = {
            "Metric": ["Price_USD_volatility_7", "Price_USD_rsi_14", "Price_USD_return_1d"],
            "XRP Mean (Std)": ["0.0785 (0.0823)", "52.53 (17.74)", "0.0058 (0.0530)"],
            "DOGE Mean (Std)": ["0.0116 (0.0114)", "50.32 (18.09)", "0.0027 (0.0527)"],
            "T-test p-value": ["< 0.001 ***", "0.104 ns", "0.448 ns"],
            "Mann-Whitney p": ["< 0.001 ***", "0.147 ns", "0.503 ns"],
            "K-S test p": ["< 0.001 ***", "0.329 ns", "0.059 ns"]
        }

        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        st.markdown("""
        - **Volatility (`Price_USD_volatility_7`)**: XRP shows significantly higher volatility than DOGE.
            - All tests (T-test, Mann-Whitney, K-S) report **p < 0.001**, indicating a **highly significant difference**.
        - **RSI (`Price_USD_rsi_14`)**: The difference in RSI between XRP and DOGE is **not statistically significant**.
            - p-values > 0.1 across all tests suggest **similar momentum behavior**.
        - **Daily Return (`Price_USD_return_1d`)**: No significant difference in daily returns.
            - High p-values (T-test: 0.448, MW: 0.503) imply **comparable daily return distributions**.
        """)

        with tab2:
            st.subheader("📢 Sentiment & Public Interest Comparison: XRP vs DOGE")

            # --- Two-column layout: Sentiment Over Time + Distribution Chart ---
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Reddit Sentiment Over Time")

                sentiment_df = pd.concat([
                    xrp_df[["date", "reddit_sentiment"]].assign(Asset="XRP"),
                    doge_df[["date", "reddit_sentiment"]].assign(Asset="DOGE")
                ])

                fig_sentiment_line = px.line(
                    sentiment_df,
                    x="date",
                    y="reddit_sentiment",
                    color="Asset",
                    labels={"reddit_sentiment": "Sentiment Score", "date": "Date"},
                    color_discrete_map={"XRP": "skyblue", "DOGE": "orange"}
                )
                st.plotly_chart(fig_sentiment_line, use_container_width=True)

            with col2:
                st.markdown("### 📊 Distribution Comparison by Sentiment Feature")

                sentiment_features = [
                    "reddit_sentiment", 
                    "reddit_sentiment_volatility_7", 
                    "search_trend"
                ]
                selected_sentiment_feature = st.selectbox("Select a sentiment feature:", sentiment_features)

                if selected_sentiment_feature in xrp_df.columns and selected_sentiment_feature in doge_df.columns:
                    xrp_data = xrp_df[selected_sentiment_feature].dropna()
                    doge_data = doge_df[selected_sentiment_feature].dropna()

                    if len(xrp_data) > 0 and len(doge_data) > 0:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.set_title(f"Distribution Comparison for {selected_sentiment_feature}", fontsize=14, fontweight='bold')

                        sns.histplot(xrp_data, kde=True, color="skyblue", label="XRP", stat="density", element="step", ax=ax)
                        sns.histplot(doge_data, kde=True, color="orange", label="DOGE", stat="density", element="step", ax=ax)

                        ax.legend()
                        ax.set_xlabel(selected_sentiment_feature)
                        ax.set_ylabel("Density")
                        st.pyplot(fig)

            # --- Statistical Comparison Table ---
            st.markdown("### 📈 Statistical Comparison: XRP vs DOGE (Sentiment & Public Interest)")

            # Fix: wrap into rows instead of columns
            sentiment_stats_rows = [
                ["reddit_sentiment", "0.2206 (0.4794)", "0.0160 (0.0840)", "< 0.001 ***", "< 0.001 ***", "< 0.001 ***"],
                ["reddit_sentiment_volatility_7", "0.4091 (0.1693)", "0.0134 (0.0548)", "< 0.001 ***", "< 0.001 ***", "< 0.001 ***"],
                ["search_trend", "23.34 (19.68)", "3.37 (3.78)", "< 0.001 ***", "< 0.001 ***", "< 0.001 ***"]
            ]

            df_sentiment = pd.DataFrame(
                sentiment_stats_rows,
                columns=["Metric", "XRP Mean (Std)", "DOGE Mean (Std)", "T-test p-value", "Mann-Whitney p", "K-S test p"]
            )

            st.dataframe(df_sentiment, use_container_width=True)

            st.markdown("""
            - **Reddit Sentiment (`reddit_sentiment`)**: XRP shows much higher sentiment fluctuations than DOGE.
            - **Volatility**: XRP has significantly more sentiment volatility.
            - **Search Trends**: XRP draws higher search interest than DOGE, statistically significant.
            """)

    with tab3:
        # Statistical Test Results at the Top
        st.subheader("Statistical Comparison of Prediction Errors")
        st.markdown("""
        **T-test p-value:** 0.0483  
        **Mann-Whitney U test p-value:** 0.0199  

        🔍 **Result**: There is a *statistically significant* difference in the prediction errors between the two models (p < 0.05).
        """)

        # Create 3 columns for the plots
        col1, col2, col3 = st.columns(3)

        # === Column 1: Bar Chart (Model Performance Metrics) ===
        with col1:
            st.subheader("Model Metrics")
            metrics_data = {
                "Metric": ["RMSE", "MAE", "R²"],
                "XRP (SVR)": [0.011219, 0.009216, 0.994419],
                "DOGE (Gradient Boosting)": [0.006483, 0.004899, 0.948458]
            }
            df_metrics = pd.DataFrame(metrics_data)

            fig1, ax1 = plt.subplots(figsize=(5, 4))
            x = range(len(df_metrics["Metric"]))
            width = 0.35

            ax1.bar([i - width/2 for i in x], df_metrics["XRP (SVR)"], width, label="XRP (SVR)", color="skyblue")
            ax1.bar([i + width/2 for i in x], df_metrics["DOGE (Gradient Boosting)"], width, label="DOGE (Gradient Boosting)", color="orange")

            ax1.set_xticks(x)
            ax1.set_xticklabels(df_metrics["Metric"])
            ax1.set_ylabel("Value")
            ax1.set_title("Performance Comparison")
            ax1.legend()
            st.pyplot(fig1)

        # === Load residuals ===
        df = pd.read_csv("xrp_doge_residuals.csv")

        # === Column 2: Boxplot ===
        with col2:
            st.subheader("Residual Boxplot")
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            sns.boxplot(data=df, x='Model', y='Residual', ax=ax2, palette=["skyblue", "orange"])
            ax2.set_title("Residual Error Distribution")
            ax2.grid(True)
            st.pyplot(fig2)

        # === Column 3: KDE Plot ===
        with col3:
            st.subheader("Residual KDE Plot")
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            sns.kdeplot(data=df[df["Model"] == "SVR (XRP)"]["Residual"], label='SVR (XRP)', fill=True, color="skyblue", ax=ax3)
            sns.kdeplot(data=df[df["Model"] == "GB (DOGE)"]["Residual"], label='GB (DOGE)', fill=True, color="orange", ax=ax3)
            ax3.set_title("Residual Distribution Comparison")
            ax3.legend()
            ax3.grid(True)
            st.pyplot(fig3)

