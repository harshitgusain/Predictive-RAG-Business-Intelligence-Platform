import streamlit as st
import pandas as pd
import plotly.express as px
import chromadb
import ollama

from sentence_transformers import SentenceTransformer
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="AI BI Copilot",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

df = pd.read_csv(
    "data/Global_Superstore2.csv",
    encoding='latin1'
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #0B1120;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

.chat-user {
    background-color: #2563EB;
    padding: 15px;
    border-radius: 15px;
    margin-top: 15px;
    color: white;
}

.chat-ai {
    background-color: #1F2937;
    padding: 15px;
    border-radius: 15px;
    margin-top: 15px;
    color: white;
    border: 1px solid #374151;
}

.kpi-card {
    background: linear-gradient(135deg,#1F2937,#111827);
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("🤖 AI BI Copilot")
st.sidebar.markdown("---")
st.sidebar.write("### 🚀 Features")
st.sidebar.write("✅ AI Executive Insights")
st.sidebar.write("✅ Dynamic Dashboards")
st.sidebar.write("✅ RAG Architecture")
st.sidebar.write("✅ ML Time-Series Forecasting")
st.sidebar.write("✅ Business Intelligence")

# ---------------------------------------------------
# KPI CARDS
# ---------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
    <h3>Total Revenue</h3>
    <h1>${round(df['Sales'].sum()/1000000,2)}M</h1>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
    <h3>Total Profit</h3>
    <h1>${round(df['Profit'].sum()/1000000,2)}M</h1>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
    <h3>Total Orders</h3>
    <h1>{df['Order ID'].nunique()}</h1>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
    <h3>Customers</h3>
    <h1>{df['Customer Name'].nunique()}</h1>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🤖 AI Business Intelligence Copilot")

query = st.text_input(
    "💬 Ask a strategic or predictive business question"
)

# ---------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer(
        'all-MiniLM-L6-v2'
    )

embedding_model = load_model()

# ---------------------------------------------------
# LOAD CHROMADB
# ---------------------------------------------------

@st.cache_resource
def load_chroma():
    client = chromadb.PersistentClient(
        path="chroma_db"
    )
    collection = client.get_or_create_collection(
        name="business_data"
    )
    return collection

collection = load_chroma()

# ---------------------------------------------------
# PROCESS QUESTION
# ---------------------------------------------------

if query:

    # USER MESSAGE
    st.markdown(
        f"""
        <div class="chat-user">
        👤 {query}
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.spinner("Analyzing business data and generating ML models..."):

        # ---------------- RAG SEARCH ---------------- #
        query_embedding = embedding_model.encode(query)

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3
        )

        context = "\n".join(results['documents'][0])

        # ---------------- AI PROMPT ---------------- #
        prompt = f"""
        You are an elite AI Business Consultant and Data Scientist.

        Use this verified business context from our database:
        {context}

        Question:
        {query}

        Provide a strategic response structured clearly with:
        - Executive Summary
        - Key Insights
        - Data-Driven Recommendations

        IMPORTANT INSTRUCTION: 
        Analyze the user's question and decide which chart would best visualize the answer. 
        At the very end of your response, output exactly ONE of the following tags:
        [CHART_REGION] - if the question asks about geography, regions, or location performance.
        [CHART_CATEGORY] - if the question asks about product categories or items.
        [CHART_PROFIT] - if the question asks about profit, loss, or margins.
        [CHART_CUSTOMER] - if the question asks about specific customers or top clients.
        [CHART_FORECAST] - if the question explicitly asks to forecast, predict, or look into future sales.
        [NO_CHART] - if the question is general and doesn't need a specific chart.
        """
        
        # ---------------- AI CHAT CALL ---------------- #
        response = ollama.chat(
            model='llama3',
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )

        answer = response['message']['content']

        # ---------------- CLEAN THE OUTPUT ---------------- #
        # Added [CHART_FORECAST] to the removal list
        tags_to_remove = ['[CHART_REGION]', '[CHART_CATEGORY]', '[CHART_PROFIT]', '[CHART_CUSTOMER]', '[CHART_FORECAST]', '[NO_CHART]']
        clean_answer = answer
        for tag in tags_to_remove:
            clean_answer = clean_answer.replace(tag, '')
            
        clean_answer = clean_answer.strip()

        # ---------------- AI RESPONSE ---------------- #
        st.markdown(
            f"""
            <div class="chat-ai">
            🤖 {clean_answer}
            </div>
            """,
            unsafe_allow_html=True
        )

    # ===================================================
    # DYNAMIC DASHBOARDS & ML FORECASTING
    # ===================================================

    # ---------------- REGION DASHBOARD ---------------- #
    if '[CHART_REGION]' in answer:
        st.subheader("📊 Regional Sales Dashboard")
        region_sales = df.groupby('Region')['Sales'].sum().reset_index()
        fig = px.bar(
            region_sales,
            x='Region',
            y='Sales',
            text_auto=True,
            title='Sales by Region'
        )
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- CATEGORY DASHBOARD ---------------- #
    if '[CHART_CATEGORY]' in answer:
        st.subheader("📊 Category Dashboard")
        category_sales = df.groupby('Category')['Sales'].sum().reset_index()
        fig = px.pie(
            category_sales,
            names='Category',
            values='Sales',
            title='Sales by Category'
        )
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- PROFIT DASHBOARD ---------------- #
    if '[CHART_PROFIT]' in answer:
        st.subheader("📈 Profit Dashboard")
        profit_data = df.groupby('Region')['Profit'].sum().reset_index()
        fig = px.bar(
            profit_data,
            x='Region',
            y='Profit',
            text_auto=True,
            title='Profit by Region'
        )
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- CUSTOMER DASHBOARD ---------------- #
    if '[CHART_CUSTOMER]' in answer:
        st.subheader("👥 Customer Dashboard")
        customer_data = df.groupby('Customer Name')['Sales'].sum().reset_index()
        customer_data = customer_data.sort_values(by='Sales', ascending=False).head(10)
        fig = px.bar(
            customer_data,
            x='Customer Name',
            y='Sales',
            title='Top 10 Customers'
        )
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)
        
    # ---------------- PREDICTIVE ML FORECAST DASHBOARD ---------------- #
    if '[CHART_FORECAST]' in answer:
        st.subheader("🔮 Predictive Revenue Horizon (Time-Series ML)")
        
        with st.spinner("Fitting Holt-Winters Exponential Smoothing Model..."):
            # 1. Prepare and clean dates
            df_time = df.copy()
            df_time['Order Date'] = pd.to_datetime(df_time['Order Date'], format='%d-%m-%Y', errors='coerce')
            df_time = df_time.dropna(subset=['Order Date'])
            
            # 2. Group by month to create a time-series
            monthly_sales = df_time.groupby(pd.Grouper(key='Order Date', freq='MS'))['Sales'].sum().reset_index()
            monthly_sales = monthly_sales.set_index('Order Date').asfreq('MS').ffill()
            
            # 3. Fit the Machine Learning Model (Accounting for Trend and Seasonality)
            model = ExponentialSmoothing(monthly_sales['Sales'], trend='add', seasonal='add', seasonal_periods=12)
            fitted_model = model.fit()
            
            # 4. Predict 6 months into the future
            forecast_steps = 6
            forecast_values = fitted_model.forecast(steps=forecast_steps)
            
            # 5. Format historical data
            history_df = monthly_sales.reset_index()
            history_df['Type'] = 'Historical Revenue'
            
            # 6. Format future predicted data
            future_dates = pd.date_range(start=monthly_sales.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
            forecast_df = pd.DataFrame({'Order Date': future_dates, 'Sales': forecast_values.values, 'Type': 'AI Predicted Target'})
            
            # 7. Combine and plot
            combined_forecast = pd.concat([history_df, forecast_df], ignore_index=True)
            
            fig = px.line(
                combined_forecast,
                x='Order Date',
                y='Sales',
                color='Type',
                title='Historical Sales vs. AI 6-Month Projection',
                color_discrete_map={'Historical Revenue': '#38bdf8', 'AI Predicted Target': '#fb7185'}
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)