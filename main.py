"""
Guldfynd Diamond Analytics & Inventory Management System

A Streamlit application for diamond market analysis and
inventory management with AI-powered integeration.
"""

import io
import json
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Guldfynd Diamond Analytics",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #f8f9fa;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #f8f9fa;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2c5aa0;
    }
    .ai-insights {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize session state
if "df" not in st.session_state:
    st.session_state.df = None
if "inventory" not in st.session_state:
    st.session_state.inventory = pd.DataFrame()
if "filters_applied" not in st.session_state:
    st.session_state.filters_applied = {}
if "ai_provider" not in st.session_state:
    st.session_state.ai_provider = "OpenAI GPT-4"
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "ai_insights" not in st.session_state:
    st.session_state.ai_insights = {}


def setup_ai_config():
    """Setup AI configuration in sidebar."""
    with st.sidebar.expander("🤖 AI Configuration", expanded=False):
        ai_provider = st.selectbox(
            "Choose AI Provider:",
            ["OpenAI GPT-4", "Google Gemini", "Local Ollama", "Disabled"],
        )
        st.session_state.ai_provider = ai_provider

        if ai_provider in ["OpenAI GPT-4", "Google Gemini"]:
            api_key = st.text_input(
                f"{ai_provider} API Key:",
                type="password",
                value=st.session_state.api_key,
                help=f"Enter your {ai_provider} API key for AI-powered insights",
            )
            st.session_state.api_key = api_key

            if not api_key:
                st.warning("⚠️ API key required for AI insights")
            else:
                st.success("✅ AI ready to generate insights")

        elif ai_provider == "Local Ollama":
            st.info("📝 Make sure Ollama is running locally on port 11434")

        elif ai_provider == "Disabled":
            st.info("AI insights disabled")


def analyze_data_statistics(df, analysis_type):
    """Extract key statistics for AI analysis."""
    stats = {"total_diamonds": len(df), "analysis_type": analysis_type}

    if analysis_type == "Carat Analysis":
        stats.update(
            {
                "avg_carat": df["carat"].mean(),
                "median_carat": df["carat"].median(),
                "carat_std": df["carat"].std(),
                "carat_range": (df["carat"].min(), df["carat"].max()),
                "carat_quartiles": df["carat"].quantile([0.25, 0.5, 0.75]).tolist(),
                "small_diamonds_pct": (df["carat"] <= 1.0).sum() / len(df) * 100,
                "large_diamonds_pct": (df["carat"] >= 2.0).sum() / len(df) * 100,
                "price_carat_correlation": df["price"].corr(df["carat"]),
            }
        )

    elif analysis_type == "Cut Analysis":
        cut_counts = df["cut"].value_counts()
        cut_prices = df.groupby("cut")["price"].mean()
        stats.update(
            {
                "cut_distribution": cut_counts.to_dict(),
                "most_common_cut": cut_counts.index[0],
                "least_common_cut": cut_counts.index[-1],
                "cut_price_avg": cut_prices.to_dict(),
                "highest_priced_cut": cut_prices.idxmax(),
                "lowest_priced_cut": cut_prices.idxmin(),
                "premium_cuts_pct": (df["cut"].isin(["Premium", "Ideal"])).sum()
                / len(df)
                * 100,
            }
        )

    elif analysis_type == "Color Analysis":
        color_counts = df["color"].value_counts().sort_index()
        color_analysis = (
            df.groupby("color").agg({"carat": "mean", "price": "mean"}).sort_index()
        )
        color_analysis["price_per_carat"] = (
            color_analysis["price"] / color_analysis["carat"]
        )

        stats.update(
            {
                "color_distribution": color_counts.to_dict(),
                "most_common_color": color_counts.idxmax(),
                "premium_colors_pct": (df["color"].isin(["D", "E", "F"])).sum()
                / len(df)
                * 100,
                "avg_carat_by_color": df.groupby("color")["carat"].mean().to_dict(),
                "avg_price_by_color": df.groupby("color")["price"].mean().to_dict(),
                "price_per_carat_by_color": color_analysis["price_per_carat"].to_dict(),
                "best_value_color": color_analysis["price_per_carat"].idxmax(),
                "color_price_paradox": (
                    df.groupby("color")["price"].mean()["J"]
                    > df.groupby("color")["price"].mean()["D"]
                    if "J" in df["color"].values and "D" in df["color"].values
                    else False
                ),
            }
        )

    elif analysis_type == "Clarity Analysis":
        clarity_counts = df["clarity"].value_counts()
        clarity_prices = df.groupby("clarity")["price"].mean()

        stats.update(
            {
                "clarity_distribution": clarity_counts.to_dict(),
                "most_common_clarity": clarity_counts.idxmax(),
                "sweet_spot_clarity_pct": (df["clarity"].isin(["SI1", "VS2"])).sum()
                / len(df)
                * 100,
                "premium_clarity_pct": (
                    df["clarity"].isin(["IF", "VVS1", "VVS2"])
                ).sum()
                / len(df)
                * 100,
                "clarity_price_avg": clarity_prices.to_dict(),
                "highest_priced_clarity": clarity_prices.idxmax(),
                "clarity_price_paradox": (
                    clarity_prices["SI2"] > clarity_prices["VVS1"]
                    if "SI2" in clarity_prices and "VVS1" in clarity_prices
                    else False
                ),
            }
        )

    elif analysis_type == "Correlation Analysis":
        if "color_numeric" in df.columns and "clarity_numeric" in df.columns:
            corr_matrix = df[
                ["price", "carat", "color_numeric", "clarity_numeric"]
            ].corr()
            stats.update(
                {
                    "price_carat_corr": corr_matrix.loc["price", "carat"],
                    "price_color_corr": corr_matrix.loc["price", "color_numeric"],
                    "price_clarity_corr": corr_matrix.loc["price", "clarity_numeric"],
                    "carat_dominates_pricing": (
                        abs(corr_matrix.loc["price", "carat"]) > 0.8
                    ),
                    "color_weak_correlation": (
                        abs(corr_matrix.loc["price", "color_numeric"]) < 0.3
                    ),
                }
            )

    return stats


def generate_ai_insights(stats, analysis_type):
    """Generate AI insights based on statistics."""
    if st.session_state.ai_provider == "Disabled":
        return None

    if (
        st.session_state.ai_provider in ["OpenAI GPT-4", "Google Gemini"]
        and not st.session_state.api_key
    ):
        return None

    # Create prompt
    prompt = f"""
You are a diamond industry analyst working for Guldfynd, a Nordic jewelry 
company expanding into diamonds. 
Analyze the following diamond market data and provide insights in this exact 
format:

From this plot we see that [describe what the data shows - key patterns, 
distributions, trends]

This has the following implications: [explain what this means for the diamond 
market, consumer behavior, pricing dynamics]

From business settings we can see that [provide specific recommendations for 
Guldfynd's Nordic market strategy, inventory decisions, pricing approach]

Analysis Type: {analysis_type}
Data Statistics: {json.dumps(stats, indent=2)}

Key Context for Guldfynd:
- Nordic market focused (Sweden, Norway, Denmark, Finland)
- Currently sells gold and silver jewelry
- Expanding into diamonds for first time  
- Target customers value quality but are price-conscious
- Need inventory strategy recommendations

Write in a professional, data-driven tone. Be specific with numbers from the 
statistics provided.
Keep each section to 2-3 sentences maximum. Focus on actionable business 
insights.
"""

    try:
        if st.session_state.ai_provider == "OpenAI GPT-4":
            return call_openai_api(prompt)
        elif st.session_state.ai_provider == "Google Gemini":
            return call_gemini_api(prompt)
        elif st.session_state.ai_provider == "Local Ollama":
            return call_ollama_api(prompt)
    except Exception as e:
        st.error(f"AI Error: {str(e)}")
        return None


def call_openai_api(prompt):
    """Call OpenAI API."""
    headers = {
        "Authorization": f"Bearer {st.session_state.api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 800,
        "temperature": 0.7,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=30,
    )

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"OpenAI API Error: {response.status_code} - {response.text}")


def call_gemini_api(prompt):
    """Call Google Gemini API."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-pro:generateContent?key={st.session_state.api_key}"
    )

    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 800, "temperature": 0.7},
    }

    response = requests.post(url, json=data, timeout=30)

    if response.status_code == 200:
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    else:
        raise Exception(f"Gemini API Error: {response.status_code} - {response.text}")


def call_ollama_api(prompt):
    """Call local Ollama API."""
    data = {
        "model": "llama3.1",
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(
        "http://localhost:11434/api/generate", json=data, timeout=60
    )

    if response.status_code == 200:
        return response.json()["response"]
    else:
        raise Exception(f"Ollama API Error: {response.status_code}")


def display_ai_insights(df, analysis_type):
    """Display AI-generated insights."""
    if st.session_state.ai_provider == "Disabled":
        return

    with st.expander("🤖 AI-Powered Business Insights", expanded=True):
        if not st.session_state.api_key and st.session_state.ai_provider in [
            "OpenAI GPT-4",
            "Google Gemini",
        ]:
            st.warning("⚠️ Please configure your API key in the sidebar for AI insights")
            return

        col1, col2 = st.columns([3, 1])

        with col1:
            if st.button(
                f"Generate AI Insights ({analysis_type})", key=f"ai_{analysis_type}"
            ):
                with st.spinner("🤖 Analyzing data and generating insights..."):
                    # Extract statistics
                    stats = analyze_data_statistics(df, analysis_type)

                    # Generate insights
                    insights = generate_ai_insights(stats, analysis_type)

                    if insights:
                        st.session_state.ai_insights[analysis_type] = insights
                        st.success("✅ AI insights generated!")
                    else:
                        st.error(
                            "Failed to generate AI insights. Please check your "
                            "API configuration."
                        )

        with col2:
            if analysis_type in st.session_state.ai_insights:
                # Option to download insights
                st.download_button(
                    "💾 Download Analysis",
                    data=st.session_state.ai_insights[analysis_type],
                    file_name=(
                        f"guldfynd_{analysis_type.lower().replace(' ', '_')}_"
                        f"analysis.txt"
                    ),
                    mime="text/plain",
                )

        # Display insights if available
        if analysis_type in st.session_state.ai_insights:
            st.markdown('<div class="ai-insights">', unsafe_allow_html=True)
            st.markdown("### 🤖 AI Business Analysis")
            st.markdown(st.session_state.ai_insights[analysis_type])
            st.markdown("</div>", unsafe_allow_html=True)


def clean_diamond_data(dirty_df):
    """Clean diamond dataset based on our previous analysis."""
    df = dirty_df.copy()

    # Drop redundant index column if present
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    # Remove exact duplicate rows
    df.drop_duplicates(inplace=True)

    # Remove rows with missing values
    df.dropna(inplace=True)

    # Remove rows with invalid dimensions
    if all(col in df.columns for col in ["x", "y", "z"]):
        df = df[(df[["x", "y", "z"]] > 0).all(axis=1)]

    # Create numeric mappings
    if "color" in df.columns:
        color_mapping = {"D": 1, "E": 2, "F": 3, "G": 4, "H": 5, "I": 6, "J": 7}
        df["color_numeric"] = df["color"].map(color_mapping)

    if "clarity" in df.columns:
        clarity_mapping = {
            "I1": 1,
            "SI2": 2,
            "SI1": 3,
            "VS2": 4,
            "VS1": 5,
            "VVS2": 6,
            "VVS1": 7,
            "IF": 8,
        }
        df["clarity_numeric"] = df["clarity"].map(clarity_mapping)

    if "cut" in df.columns:
        # Map to GIA standards
        cut_mapping = {
            "Fair": "Poor",
            "Good": "Good",
            "Very Good": "Very Good",
            "Premium": "Excellent",
            "Ideal": "Excellent",
        }
        df["cut_gia"] = df["cut"].map(cut_mapping)

    return df


def load_data():
    """Data loading interface."""
    st.markdown(
        '<div class="section-header">📁 Data Upload</div>', unsafe_allow_html=True
    )

    upload_option = st.radio(
        "Choose data source:", ["Upload File", "Connect to Database"]
    )

    if upload_option == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload your diamond dataset",
            type=["csv", "xlsx", "xls"],
            help="Supported formats: CSV, Excel (xlsx, xls)",
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                return df
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                return None

    elif upload_option == "Connect to Database":
        st.info("🚧 Database connection feature coming soon!")
        return None
    return None


def create_overview_metrics(df):
    """Create overview metrics."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Diamonds", f"{len(df):,}")
    with col2:
        st.metric("Avg Price", f"${df['price'].mean():,.0f}")
    with col3:
        st.metric("Avg Carat", f"{df['carat'].mean():.2f}")
    with col4:
        st.metric("Price Range", f"${df['price'].min():,} - ${df['price'].max():,}")


def create_carat_analysis(df):
    """Carat weight analysis plots."""
    st.markdown(
        '<div class="section-header">💎 Carat Weight Analysis</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        # Carat distribution
        fig1 = px.histogram(
            df,
            x="carat",
            nbins=30,
            title="Distribution of Diamond Carat Weights",
            labels={"carat": "Carat Weight", "count": "Frequency"},
        )
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Cumulative distribution
        sorted_carats = np.sort(df["carat"])
        cumulative_pct = np.arange(1, len(sorted_carats) + 1) / len(sorted_carats) * 100

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=sorted_carats, y=cumulative_pct, mode="lines", name="Cumulative %"
            )
        )
        fig2.add_hline(
            y=80, line_dash="dash", line_color="red", annotation_text="80% threshold"
        )
        fig2.add_vline(
            x=1.2,
            line_dash="dash",
            line_color="red",
            annotation_text="1.2 carat cutoff",
        )
        fig2.update_layout(
            title="Cumulative Distribution - Market Capture",
            xaxis_title="Carat Weight",
            yaxis_title="Cumulative Percentage (%)",
            height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Price vs Carat
    fig3 = px.scatter(
        df,
        x="carat",
        y="price",
        opacity=0.6,
        title="Price vs Carat Weight Relationship",
        labels={"carat": "Carat Weight", "price": "Price ($)"},
    )
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

    # AI Insights
    display_ai_insights(df, "Carat Analysis")


def create_cut_analysis(df):
    """Cut quality analysis."""
    st.markdown(
        '<div class="section-header">✂️ Cut Quality Analysis</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        # Cut distribution
        cut_counts = df["cut"].value_counts()
        fig1 = px.bar(
            x=cut_counts.index,
            y=cut_counts.values,
            title="Distribution of Diamond Cuts",
            labels={"x": "Cut Quality", "y": "Count"},
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Average price by cut
        avg_price_by_cut = df.groupby("cut")["price"].mean().sort_values()
        fig2 = px.bar(
            x=avg_price_by_cut.index,
            y=avg_price_by_cut.values,
            title="Average Price by Cut Quality",
            labels={"x": "Cut Quality", "y": "Average Price ($)"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    # AI Insights
    display_ai_insights(df, "Cut Analysis")


def create_color_analysis(df):
    """Color grade analysis."""
    st.markdown(
        '<div class="section-header">🌈 Color Grade Analysis</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        # Color distribution
        color_counts = df["color"].value_counts().sort_index()
        fig1 = px.bar(
            x=color_counts.index,
            y=color_counts.values,
            title="Distribution of Diamond Colors",
            labels={"x": "Color Grade (D=Best)", "y": "Count"},
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Price per carat by color
        color_analysis = (
            df.groupby("color").agg({"carat": "mean", "price": "mean"}).sort_index()
        )
        color_analysis["price_per_carat"] = (
            color_analysis["price"] / color_analysis["carat"]
        )

        fig2 = px.bar(
            x=color_analysis.index,
            y=color_analysis["price_per_carat"],
            title="Price Per Carat by Color Grade",
            labels={"x": "Color Grade (D=Best)", "y": "Price per Carat ($)"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    # AI Insights
    display_ai_insights(df, "Color Analysis")


def create_clarity_analysis(df):
    """Clarity grade analysis."""
    st.markdown(
        '<div class="section-header">🔍 Clarity Grade Analysis</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        # Clarity distribution
        clarity_counts = df["clarity"].value_counts()
        fig1 = px.bar(
            x=clarity_counts.index,
            y=clarity_counts.values,
            title="Distribution of Diamond Clarity",
            labels={"x": "Clarity (IF=Best)", "y": "Count"},
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Price distribution by clarity
        fig2 = px.box(
            df,
            x="clarity",
            y="price",
            title="Price Distribution by Clarity Grade",
            labels={"clarity": "Clarity Grade", "price": "Price ($)"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    # AI Insights
    display_ai_insights(df, "Clarity Analysis")


def create_correlation_analysis(df):
    """Correlation analysis."""
    st.markdown(
        '<div class="section-header">📊 Correlation Analysis</div>',
        unsafe_allow_html=True,
    )

    # Create correlation matrix
    if "color_numeric" in df.columns and "clarity_numeric" in df.columns:
        corr_data = df[["price", "carat", "color_numeric", "clarity_numeric"]].corr()

        fig = px.imshow(
            corr_data,
            title="Correlation Matrix: Diamond Characteristics",
            labels={"x": "Variables", "y": "Variables", "color": "Correlation"},
            text_auto=True,
            aspect="auto",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Business insights
        price_carat_corr = corr_data.loc["price", "carat"]
        price_color_corr = corr_data.loc["price", "color_numeric"]
        price_clarity_corr = corr_data.loc["price", "clarity_numeric"]

        st.info(
            f"""
        **Key Insights:**
        - Carat vs Price correlation: {price_carat_corr:.3f} (Very Strong)
        - Color vs Price correlation: {price_color_corr:.3f} (Weak)
        - Clarity vs Price correlation: {price_clarity_corr:.3f} (Moderate)
        
        **Business Implication:** Size (carat) dominates pricing over color and 
        clarity quality.
        """
        )

    # AI Insights
    display_ai_insights(df, "Correlation Analysis")


def create_filters(df):
    """Create dynamic filtering interface."""
    st.markdown(
        '<div class="section-header">🔧 Dynamic Filters</div>', unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        # Price filter
        price_range = st.slider(
            "Price Range ($)",
            min_value=int(df["price"].min()),
            max_value=int(df["price"].max()),
            value=(int(df["price"].min()), int(df["price"].max())),
            step=100,
        )

    with col2:
        # Carat filter
        carat_range = st.slider(
            "Carat Range",
            min_value=float(df["carat"].min()),
            max_value=float(df["carat"].max()),
            value=(float(df["carat"].min()), float(df["carat"].max())),
            step=0.1,
        )

    with col3:
        # Color filter
        color_options = st.multiselect(
            "Color Grades",
            options=sorted(df["color"].unique()),
            default=sorted(df["color"].unique()),
        )

    col4, col5 = st.columns(2)

    with col4:
        # Clarity filter
        clarity_options = st.multiselect(
            "Clarity Grades",
            options=df["clarity"].unique(),
            default=df["clarity"].unique(),
        )

    with col5:
        # Cut filter
        cut_options = st.multiselect(
            "Cut Grades", options=df["cut"].unique(), default=df["cut"].unique()
        )

    # Apply filters
    filtered_df = df[
        (df["price"] >= price_range[0])
        & (df["price"] <= price_range[1])
        & (df["carat"] >= carat_range[0])
        & (df["carat"] <= carat_range[1])
        & (df["color"].isin(color_options))
        & (df["clarity"].isin(clarity_options))
        & (df["cut"].isin(cut_options))
    ]

    st.info(
        f"Filtered results: {len(filtered_df):,} diamonds out of " f"{len(df):,} total"
    )

    return filtered_df


def inventory_management(filtered_df):
    """Inventory management system."""
    st.markdown(
        '<div class="section-header">📦 Inventory Management</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Available Diamonds")

        # Display filtered diamonds
        if len(filtered_df) > 0:
            display_df = filtered_df[
                ["carat", "cut", "color", "clarity", "price"]
            ].head(100)

            # Add selection column
            def format_diamond(x):
                return (
                    f"Diamond {x}: {display_df.loc[x, 'carat']:.2f}ct, "
                    f"{display_df.loc[x, 'color']}-{display_df.loc[x, 'clarity']}, "
                    f"${display_df.loc[x, 'price']:,}"
                )

            selected_indices = st.multiselect(
                "Select diamonds to add to inventory:",
                options=list(display_df.index),
                format_func=format_diamond,
            )

            st.dataframe(display_df)

            # Add selected diamonds to inventory
            if st.button("Add Selected to Inventory") and selected_indices:
                selected_diamonds = filtered_df.loc[selected_indices]
                st.session_state.inventory = pd.concat(
                    [st.session_state.inventory, selected_diamonds], ignore_index=True
                )
                st.success(f"Added {len(selected_indices)} diamonds to inventory!")

            # Add all filtered diamonds
            if st.button("Add All Filtered to Inventory"):
                st.session_state.inventory = pd.concat(
                    [st.session_state.inventory, filtered_df], ignore_index=True
                )
                st.success(f"Added {len(filtered_df)} diamonds to inventory!")

    with col2:
        st.subheader("Manual Diamond Entry")

        with st.form("add_diamond"):
            manual_carat = st.number_input(
                "Carat", min_value=0.1, max_value=10.0, value=1.0, step=0.1
            )
            manual_cut = st.selectbox(
                "Cut", options=["Ideal", "Premium", "Very Good", "Good", "Fair"]
            )
            manual_color = st.selectbox(
                "Color", options=["D", "E", "F", "G", "H", "I", "J"]
            )
            manual_clarity = st.selectbox(
                "Clarity",
                options=["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"],
            )
            manual_price = st.number_input(
                "Price ($)", min_value=100, max_value=100000, value=5000, step=100
            )

            if st.form_submit_button("Add Manual Diamond"):
                new_diamond = pd.DataFrame(
                    {
                        "carat": [manual_carat],
                        "cut": [manual_cut],
                        "color": [manual_color],
                        "clarity": [manual_clarity],
                        "price": [manual_price],
                    }
                )
                st.session_state.inventory = pd.concat(
                    [st.session_state.inventory, new_diamond], ignore_index=True
                )
                st.success("Manual diamond added to inventory!")


def display_inventory():
    """Display and manage inventory."""
    st.markdown(
        '<div class="section-header">📋 Current Inventory</div>', unsafe_allow_html=True
    )

    if len(st.session_state.inventory) > 0:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Diamonds", len(st.session_state.inventory))
        with col2:
            total_value = st.session_state.inventory["price"].sum()
            st.metric("Total Value", f"${total_value:,}")
        with col3:
            avg_price = st.session_state.inventory["price"].mean()
            st.metric("Avg Price", f"${avg_price:,.0f}")

        # Display inventory
        st.dataframe(st.session_state.inventory)

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Clear Inventory"):
                st.session_state.inventory = pd.DataFrame()
                st.success("Inventory cleared!")

        with col2:
            # Download inventory
            csv = st.session_state.inventory.to_csv(index=False)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="Download Inventory (CSV)",
                data=csv,
                file_name=f"guldfynd_inventory_{timestamp}.csv",
                mime="text/csv",
            )

        with col3:
            # Download as Excel
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                st.session_state.inventory.to_excel(
                    writer, sheet_name="Inventory", index=False
                )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="Download Inventory (Excel)",
                data=buffer.getvalue(),
                file_name=f"guldfynd_inventory_{timestamp}.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
            )
    else:
        st.info(
            "No diamonds in inventory yet. Use the filters and selection tools "
            "above to add diamonds."
        )


def main():
    """Main application."""
    st.markdown(
        '<div class="main-header">💎 Guldfynd Diamond Analytics & '
        "Inventory Management</div>",
        unsafe_allow_html=True,
    )

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a section:",
        ["Data Upload", "Overview", "Analysis", "Inventory Management"],
    )

    # AI Configuration
    setup_ai_config()

    if page == "Data Upload":
        df = load_data()
        if df is not None:
            st.session_state.df = clean_diamond_data(df)
            st.success("✅ Data cleaned and ready for analysis!")

            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(st.session_state.df.head())

            st.subheader("Data Quality Report")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(st.session_state.df))
            with col2:
                st.metric("Columns", len(st.session_state.df.columns))
            with col3:
                missing_values = st.session_state.df.isnull().sum().sum()
                st.metric("Missing Values", missing_values)

    elif page == "Overview" and st.session_state.df is not None:
        create_overview_metrics(st.session_state.df)

        # Quick insights
        st.markdown(
            '<div class="section-header">📈 Quick Insights</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            # Price distribution
            fig = px.histogram(
                st.session_state.df, x="price", nbins=50, title="Price Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Carat vs Price
            fig = px.scatter(
                st.session_state.df,
                x="carat",
                y="price",
                opacity=0.6,
                title="Price vs Carat",
            )
            st.plotly_chart(fig, use_container_width=True)

    elif page == "Analysis" and st.session_state.df is not None:
        analysis_type = st.selectbox(
            "Select Analysis Type:",
            [
                "Carat Analysis",
                "Cut Analysis",
                "Color Analysis",
                "Clarity Analysis",
                "Correlation Analysis",
            ],
        )

        if analysis_type == "Carat Analysis":
            create_carat_analysis(st.session_state.df)
        elif analysis_type == "Cut Analysis":
            create_cut_analysis(st.session_state.df)
        elif analysis_type == "Color Analysis":
            create_color_analysis(st.session_state.df)
        elif analysis_type == "Clarity Analysis":
            create_clarity_analysis(st.session_state.df)
        elif analysis_type == "Correlation Analysis":
            create_correlation_analysis(st.session_state.df)

    elif page == "Inventory Management" and st.session_state.df is not None:
        # Apply filters
        filtered_df = create_filters(st.session_state.df)

        # Inventory management
        inventory_management(filtered_df)

        # Display current inventory
        display_inventory()

    elif st.session_state.df is None:
        st.warning("⚠️ Please upload data first using the 'Data Upload' section.")
        st.info(
            "👈 Use the sidebar to navigate to 'Data Upload' and load your "
            "diamond dataset."
        )


if __name__ == "__main__":
    main()
