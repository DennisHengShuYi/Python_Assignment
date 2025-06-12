import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import numpy as np


# Page configuration
st.set_page_config(
    page_title="🚀 Memecoin Mania Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .crypto-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
    }
    
    .section-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .sidebar .stSelectbox {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    .chart-container {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .highlight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1rem;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Animated title
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h1 style='background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #f5576c);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               background-clip: text;
               font-size: 3rem;
               font-weight: bold;
               margin-bottom: 0.5rem;'>
        🚀 Memecoin Mania Dashboard
    </h1>
    <p style='font-size: 1.2rem; color: #666; margin-top: 0;'>
        Comprehensive Analysis & Real-time Insights
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown("## 🎯 Navigation")
page = st.sidebar.selectbox(
    "Choose Analysis Section",
    ["📊 Overview", "💰 Market Data", "📈 Sentiment Analysis", "📉 Trend Analysis", "🔍 Deep Dive"]
)

# === OVERVIEW PAGE ===
if page == "📊 Overview":
    # Hero metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🪙 Coins Tracked</h3>
            <h1>4</h1>
            <p>DOGE, SHIB, XRP, ADA</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📅 Data Period</h3>
            <h1>366</h1>
            <p>Days of Analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📰 News Articles</h3>
            <h1>100</h1>
            <p>Sentiment Sources</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>👾 Reddit Posts</h3>
            <h1>318</h1>
            <p>Community Insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # What are Memecoins section with enhanced styling
    st.markdown('<div class="section-header">🪙 What Are Memecoins?</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight-box">
        <p style="font-size: 1.1rem; line-height: 1.6;">
            Memecoins are cryptocurrency tokens inspired by internet memes, jokes, or viral trends. 
            Unlike traditional cryptocurrencies, they often lack fundamental utility and are primarily 
            driven by community hype, social media buzz, and speculative trading.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="crypto-card">
            <h3>📌 Key Characteristics</h3>
            <ul style="text-align: left;">
                <li>🎭 Based on memes/pop culture</li>
                <li>💸 Low individual price</li>
                <li>🚫 Limited real-world utility</li>
                <li>📱 Social media driven</li>
                <li>⚡ High volatility</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="crypto-card">
            <h3>🚀 Popularity Drivers</h3>
            <ul style="text-align: left;">
                <li>🐦 Celebrity endorsements</li>
                <li>👥 Strong communities</li>
                <li>📈 Quick profit potential</li>
                <li>🎯 FOMO (Fear of Missing Out)</li>
                <li>💰 Accessibility & low cost</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="crypto-card">
            <h3>🌟 Top Examples</h3>
            <div style="text-align: left;">
                <p><strong>🐶 DOGE</strong><br/>The original meme coin</p>
                <p><strong>🐕 SHIB</strong><br/>"Dogecoin killer"</p>
                <p><strong>🐸 PEPE</strong><br/>Meme frog phenomenon</p>
                <p><strong>💎 XRP</strong><br/>Utility-focused alternative</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Market share visualization
    st.markdown('<div class="section-header">📊 Market Composition</div>', unsafe_allow_html=True)
    
    # Sample data for demonstration
    market_data = {
        'Coin': ['DOGE', 'SHIB', 'PEPE', 'Others'],
        'Market Share': [45, 22, 15, 18],
        'Colors': ['#FFA500', '#FF6B6B', '#4ECDC4', '#45B7D1']
    }
    
    fig_pie = px.pie(
        values=market_data['Market Share'],
        names=market_data['Coin'],
        title="Memecoin Market Share Distribution",
        color_discrete_sequence=market_data['Colors'],
        hole=0.4
    )
    fig_pie.update_layout(
        title_font_size=20,
        font=dict(size=14),
        showlegend=True,
        height=400
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# === MARKET DATA PAGE ===
elif page == "💰 Market Data":
    st.markdown('<div class="section-header">💰 Market Data Overview</div>', unsafe_allow_html=True)
    
    # Data source tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔹 Ripple (XRP)", "🐶 All Memecoins", "👾 Reddit Data", "📰 News & Trends"])
    
    with tab1:
        st.markdown("""
        <div class="highlight-box">
            <h3>🔹 Ripple Market Data</h3>
            <p><strong>Source:</strong> CoinGecko API | <strong>Duration:</strong> May 2024 – June 2025 (366 days)</p>
            <p><strong>Metrics:</strong> Price, Market Cap, Volume, Rank, Supply, All-Time Highs</p>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            ripple_df = pd.read_csv("ripple_market_data.csv")
            ripple_df["date"] = pd.to_datetime(ripple_df["Timestamp"]).dt.date
            
            col1, col2 = st.columns([2, 1])
            with col1:
                # Interactive price chart
                fig = px.line(
                    ripple_df, 
                    x="date", 
                    y="Price_USD",
                    title="XRP Price Movement",
                    color_discrete_sequence=['#00D4FF']
                )
                fig.update_layout(
                    title_font_size=18,
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if len(ripple_df) > 0:
                    latest_price = ripple_df["Price_USD"].iloc[-1]
                    price_change = ripple_df["Price_USD"].pct_change().iloc[-1] * 100
                    
                    st.metric(
                        label="Current Price",
                        value=f"${latest_price:.4f}",
                        delta=f"{price_change:.2f}%"
                    )
            
            with st.expander("📊 View Raw Data"):
                st.dataframe(ripple_df.head(10), use_container_width=True)
                
        except FileNotFoundError:
            st.error("⚠️ ripple_market_data.csv not found. Please ensure the file is available.")
    
    with tab2:
        st.markdown("""
        <div class="highlight-box">
            <h3>💹 Multi-Coin Analysis</h3>
            <p><strong>Coins:</strong> Bitcoin, Shiba Inu, Dogecoin, Cardano</p>
            <p><strong>Features:</strong> Price, Volume, Market Cap, Daily Returns, Rolling Statistics</p>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            memecoins_df = pd.read_csv("all_memecoins_market_data.csv")
            memecoins_df["date"] = pd.to_datetime(memecoins_df["Timestamp"]).dt.date
            
            # Coin selector
            selected_coin = st.selectbox(
                "🎯 Select Cryptocurrency", 
                memecoins_df["Symbol"].unique(),
                key="memecoin_selector"
            )
            
            filtered_df = memecoins_df[memecoins_df["Symbol"] == selected_coin]
            
            col1, col2 = st.columns(2)
            with col1:
                # Price chart
                fig_price = px.line(
                    filtered_df, 
                    x="date", 
                    y="Price_USD",
                    title=f"{selected_coin} Price Trend",
                    color_discrete_sequence=['#FF6B6B']
                )
                st.plotly_chart(fig_price, use_container_width=True)
            
            with col2:
                # Volume chart
                fig_volume = px.bar(
                    filtered_df.tail(30), 
                    x="date", 
                    y="Total_Volume_USD",
                    title=f"{selected_coin} Trading Volume (Last 30 Days)",
                    color_discrete_sequence=['#4ECDC4']
                )
                st.plotly_chart(fig_volume, use_container_width=True)
                
        except FileNotFoundError:
            st.error("⚠️ all_memecoins_market_data.csv not found.")
    
    with tab3:
        st.markdown("""
        <div class="highlight-box">
            <h3>👾 Reddit Community Insights</h3>
            <p><strong>Platform:</strong> Reddit API | <strong>Posts:</strong> 318 XRP-related discussions</p>
            <p><strong>Timeframe:</strong> 4 months | <strong>Metrics:</strong> Upvotes, Comments, Sentiment</p>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            reddit_df = pd.read_csv("xrp_reddit_data.csv")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_upvotes = reddit_df["post_upvotes"].mean()
                st.metric("Avg Upvotes", f"{avg_upvotes:.0f}")
            with col2:
                total_comments = reddit_df["num_comments"].sum()
                st.metric("Total Comments", f"{total_comments:,}")
            with col3:
                total_posts = len(reddit_df)
                st.metric("Total Posts", f"{total_posts}")
            
            # Engagement distribution
            fig_engagement = px.scatter(
                reddit_df,
                x="post_upvotes",
                y="num_comments",
                title="Reddit Post Engagement Analysis",
                color="post_upvotes",
                size="num_comments",
                hover_data=["created_at"]
            )
            st.plotly_chart(fig_engagement, use_container_width=True)
            
        except FileNotFoundError:
            st.error("⚠️ xrp_reddit_data.csv not found.")
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="highlight-box">
                <h3>📰 News Headlines</h3>
                <p><strong>Articles:</strong> 100 memecoin news pieces</p>
                <p><strong>Sources:</strong> NewsAPI aggregation</p>
                <p><strong>Coverage:</strong> 1 year analysis period</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="highlight-box">
                <h3>🔍 Google Trends</h3>
                <p><strong>Keywords:</strong> dogecoin, shiba inu, bitcoin, cardano</p>
                <p><strong>Frequency:</strong> Weekly data points</p>
                <p><strong>Period:</strong> 12 months trending data</p>
            </div>
            """, unsafe_allow_html=True)
        
        try:
            trends_df = pd.read_csv("search_trends_extended.csv")
            trends_df["date"] = pd.to_datetime(trends_df["date"])
            
            # Multi-line trends chart
            fig_trends = go.Figure()
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA500']
            coins = ["dogecoin", "shiba inu", "bitcoin", "cardano"]
            
            for i, coin in enumerate(coins):
                if coin in trends_df.columns:
                    fig_trends.add_trace(go.Scatter(
                        x=trends_df["date"],
                        y=trends_df[coin],
                        mode='lines+markers',
                        name=coin.title(),
                        line=dict(color=colors[i], width=3),
                        marker=dict(size=6)
                    ))
            
            fig_trends.update_layout(
                title="Google Search Trends Comparison",
                title_font_size=18,
                xaxis_title="Date",
                yaxis_title="Search Interest",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig_trends, use_container_width=True)
            
        except FileNotFoundError:
            st.error("⚠️ search_trends_extended.csv not found.")

# === SENTIMENT ANALYSIS PAGE ===
elif page == "📈 Sentiment Analysis":
    st.markdown('<div class="section-header">📈 Sentiment Analysis: DOGE vs XRP</div>', unsafe_allow_html=True)
    
    try:
        # Load data with error handling
        @st.cache_data
        def load_sentiment_data():
            doge_df = pd.read_csv("doge_adjusted_market_sentiment_search.csv", parse_dates=["date"])
            xrp_df = pd.read_csv("xrp_sentiment_dashboard_ready.csv", parse_dates=["date"])
            return doge_df, xrp_df
        
        doge_df, xrp_df = load_sentiment_data()
        
        # Filter Dogecoin data
        doge_filtered = doge_df[
            (doge_df['date'] >= pd.to_datetime("2025-04-21")) &
            (doge_df['date'] < pd.to_datetime("2025-05-29"))
        ]
        
        # Comparison metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            doge_avg_sentiment = doge_filtered['reddit_sentiment'].mean()
            st.metric("🐶 DOGE Avg Sentiment", f"{doge_avg_sentiment:.3f}")
        
        with col2:
            xrp_avg_sentiment = xrp_df['reddit_sentiment'].mean()
            st.metric("💠 XRP Avg Sentiment", f"{xrp_avg_sentiment:.3f}")
        
        with col3:
            doge_search_trend = doge_filtered['search_trend'].mean()
            st.metric("🐶 DOGE Search Interest", f"{doge_search_trend:.1f}")
        
        with col4:
            xrp_search_trend = xrp_df['search_trend'].mean()
            st.metric("💠 XRP Search Interest", f"{xrp_search_trend:.1f}")
        
        # Side-by-side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🐶 Dogecoin Analysis")
            
            # DOGE sentiment over time
            doge_daily = doge_filtered.groupby('date').agg({
                'reddit_sentiment': 'mean',
                'search_trend': 'mean'
            }).reset_index()
            
            fig_doge = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Reddit Sentiment', 'Search Trend'),
                vertical_spacing=0.12
            )
            
            fig_doge.add_trace(
                go.Scatter(
                    x=doge_daily['date'],
                    y=doge_daily['reddit_sentiment'],
                    mode='lines+markers',
                    name='Reddit Sentiment',
                    line=dict(color='#FFA500', width=3)
                ),
                row=1, col=1
            )
            
            fig_doge.add_trace(
                go.Scatter(
                    x=doge_daily['date'],
                    y=doge_daily['search_trend'],
                    mode='lines+markers',
                    name='Search Trend',
                    line=dict(color='#FF6B6B', width=3)
                ),
                row=2, col=1
            )
            
            fig_doge.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_doge, use_container_width=True)
            
            # DOGE distribution
            fig_doge_dist = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Sentiment Distribution', 'Search Trend Distribution')
            )
            
            fig_doge_dist.add_trace(
                go.Histogram(
                    x=doge_filtered['reddit_sentiment'].dropna(),
                    nbinsx=20,
                    name='Reddit Sentiment',
                    marker_color='#FFA500',
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            fig_doge_dist.add_trace(
                go.Histogram(
                    x=doge_filtered['search_trend'].dropna(),
                    nbinsx=20,
                    name='Search Trend',
                    marker_color='#FF6B6B',
                    opacity=0.7
                ),
                row=1, col=2
            )
            
            fig_doge_dist.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_doge_dist, use_container_width=True)
        
        with col2:
            st.markdown("### 💠 XRP Analysis")
            
            # XRP sentiment over time
            xrp_daily = xrp_df.groupby(xrp_df['date'].dt.date).agg({
                "reddit_sentiment": "mean",
                "search_trend": "mean"
            }).reset_index()
            
            fig_xrp = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Reddit Sentiment', 'Search Trend'),
                vertical_spacing=0.12
            )
            
            fig_xrp.add_trace(
                go.Scatter(
                    x=xrp_daily['date'],
                    y=xrp_daily['reddit_sentiment'],
                    mode='lines+markers',
                    name='Reddit Sentiment',
                    line=dict(color='#00D4FF', width=3)
                ),
                row=1, col=1
            )
            
            fig_xrp.add_trace(
                go.Scatter(
                    x=xrp_daily['date'],
                    y=xrp_daily['search_trend'],
                    mode='lines+markers',
                    name='Search Trend',
                    line=dict(color='#4ECDC4', width=3)
                ),
                row=2, col=1
            )
            
            fig_xrp.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_xrp, use_container_width=True)
            
            # XRP distribution
            fig_xrp_dist = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Sentiment Distribution', 'Search Trend Distribution')
            )
            
            fig_xrp_dist.add_trace(
                go.Histogram(
                    x=xrp_df['reddit_sentiment'].dropna(),
                    nbinsx=20,
                    name='Reddit Sentiment',
                    marker_color='#00D4FF',
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            fig_xrp_dist.add_trace(
                go.Histogram(
                    x=xrp_df['search_trend'].dropna(),
                    nbinsx=20,
                    name='Search Trend',
                    marker_color='#4ECDC4',
                    opacity=0.7
                ),
                row=1, col=2
            )
            
            fig_xrp_dist.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_xrp_dist, use_container_width=True)
        
        # Correlation analysis
        st.markdown("### 🔄 Cross-Correlation Analysis")
        
        correlation_data = pd.DataFrame({
            'DOGE Sentiment': doge_filtered['reddit_sentiment'].dropna(),
            'DOGE Search': doge_filtered['search_trend'].dropna()
        })
        
        fig_corr = px.scatter(
            correlation_data,
            x='DOGE Sentiment',
            y='DOGE Search',
            title='DOGE: Sentiment vs Search Interest Correlation',
            trendline='ols',
            color_discrete_sequence=['#FFA500']
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
    except FileNotFoundError as e:
        st.error(f"⚠️ Data file not found: {e}")
    except Exception as e:
        st.error(f"⚠️ Error loading sentiment data: {e}")

# === TREND ANALYSIS PAGE ===
elif page == "📉 Trend Analysis":
    st.markdown('<div class="section-header">📉 Advanced Trend Analysis</div>', unsafe_allow_html=True)
    
    # Feature Engineering Overview
    st.markdown("""
    <div class="highlight-box">
        <h3>🔧 Feature Engineering Pipeline</h3>
        <p>Our analysis incorporates multiple sophisticated feature categories to capture market dynamics comprehensively.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature categories
    feature_data = {
        "Feature Category": [
            "📈 Lag Features",
            "📊 Technical Indicators", 
            "💹 Returns & Volatility",
            "💭 Sentiment Features",
            "🔍 Search Trend Features",
            "📅 Temporal Features"
        ],
        "Examples": [
            "Price_lag1, sentiment_lag1, volume_lag3",
            "EMA_7, RSI_14, MACD, Bollinger Bands",
            "daily_return, volatility_7d, sharpe_ratio",
            "sentiment_momentum, rolling_mean_3d",
            "search_lag1, search_volatility, trend_slope",
            "day_of_week, month, quarter, is_weekend"
        ],
        "Purpose": [
            "Capture historical momentum patterns",
            "Identify technical trading signals",
            "Measure risk and price fluctuations", 
            "Track community sentiment shifts",
            "Model public interest dynamics",
            "Account for temporal seasonality"
        ]
    }
    
    feature_df = pd.DataFrame(feature_data)
    st.dataframe(feature_df, use_container_width=True)
    
    # Interactive trend analysis
    st.markdown("### 📊 Interactive Price & Volume Analysis")
    
    # try:
    #     # Sample trend data (replace with actual data loading)
    #     doge_df = pd.read_csv("doge_adjusted_market_sentiment_search.csv")
        
    #     analysis_type = st.selectbox(
    #         "Choose Analysis Type",
    #         ["Price Trends", "Volume Analysis", "Volatility Patterns", "Technical Indicators"]
    #     )
        
    #     if analysis_type == "Price Trends":
    #         fig = px.line(
    #             doge_df,
    #             x="Timestamp",
    #             y="Price_USD",
    #             title="🐶 Dogecoin Price Movement with Trend Analysis",
    #             color_discrete_sequence=['#FFA500']
    #         )
            
    #         # Add moving averages
    #         if len(doge_df) > 7:
    #             doge_df['MA_7'] = doge_df['Price_USD'].rolling(window=7).mean()
    #             doge_df['MA_30'] = doge_df['Price_USD'].rolling(window=30).mean()
                
    #             fig.add_trace(go.Scatter(
    #                 x=doge_df["Timestamp"],
    #                 y=doge_df['MA_7'],
    #                 mode='lines',
    #                 name='7-Day MA',
    #                 line=dict(color='red', dash='dash')
    #             ))
                
    #             fig.add_trace(go.Scatter(
    #                 x=doge_df["Timestamp"],
    #                 y=doge_df['MA_30'],
    #                 mode='lines',
    #                 name='30-Day MA',
    #                 line=dict(color='blue', dash='dot')
    #             ))
            
    #         st.plotly_chart(fig, use_container_width=True)
        
    #     elif analysis_type == "Volume Analysis":
    #         fig = px.bar(
    #             doge_df.tail(50),
    #             x="Timestamp",
    #             y="Total_Volume_USD",
    #             title="🐶 Dogecoin Trading Volume (Last 50 Days)",
    #             color="Total_Volume_USD",
    #             color_continuous_scale="Viridis"
    #         )
    #         st.plotly_chart(fig, use_container_width=True)
        
    #     elif analysis_type == "Volatility Patterns":
    #         # Calculate volatility
    #         doge_df['daily_return'] = doge_df['Price_USD'].pct_change()
    #         doge_df['volatility_7d'] = doge_df['daily_return'].rolling(window=7).std()
            
    #         fig = px.line(
    #             doge_df,
    #             x="Timestamp",
    #             y="volatility_7d",
    #             title="🐶 Dogecoin 7-Day Rolling Volatility",
    #             color_discrete_sequence=['#FF6B6B']
    #         )
    #         st.plotly_chart(fig, use_container_width=True)
        
    #     #elif analysis_type == "Technical Indicators":
    #         # RSI calculation (simplified)
            