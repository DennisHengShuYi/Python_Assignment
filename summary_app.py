import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    doge_df = pd.read_csv("doge_market_sentiment_search.csv", parse_dates=["date"])
    xrp_df = pd.read_csv("xrp_sentiment_dashboard_ready.csv", parse_dates=["date"])
    return doge_df, xrp_df

doge_df, xrp_df = load_data()

# Layout columns
col1, col2 = st.columns(2)

# --- DOGE SECTION ---
with col1:
    st.subheader("🐶 Dogecoin ")

    # Line chart with expander
    with st.expander("📈 Select Dogecoin Line Chart View"):
        doge_option = st.radio("Choose metric to display over time (Dogecoin):", ["Reddit Sentiment", "Search Trend"])

        doge_daily = doge_df.groupby(doge_df['date'].dt.date).agg({
            "reddit_sentiment": "mean",
            "search_trend": "mean"
        }).reset_index()

        if doge_option == "Reddit Sentiment":
            st.line_chart(doge_daily.rename(columns={"date": "Date", "reddit_sentiment": "Value"}).set_index("Date")["Value"])
        else:
            st.line_chart(doge_daily.rename(columns={"date": "Date", "search_trend": "Value"}).set_index("Date")["Value"])

    # Distribution chart with expander
    with st.expander("📊 Dogecoin Distribution View"):
        doge_dist_option = st.radio("Choose distribution to display (Dogecoin):", ["Reddit Sentiment", "Search Trend"])

        fig, ax = plt.subplots()
        if doge_dist_option == "Reddit Sentiment":
            sns.histplot(doge_df['reddit_sentiment'].dropna(), bins=30, kde=True, ax=ax, color="orange")
            ax.set_title("Distribution of Reddit Sentiment (Dogecoin)")
        else:
            sns.histplot(doge_df['search_trend'].dropna(), bins=30, kde=True, ax=ax, color="green")
            ax.set_title("Distribution of Search Trend (Dogecoin)")
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