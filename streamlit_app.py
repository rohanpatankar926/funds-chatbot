import streamlit as st
import os
from fund_chatbot import FundDataAnalyzer, FundChatbot
import pandas as pd

st.set_page_config(
    page_title="Fund Data Chatbot",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "analyzer" not in st.session_state:
    try:
        st.session_state.analyzer = FundDataAnalyzer()
        st.session_state.chatbot_initialized = True
    except Exception as e:
        st.session_state.chatbot_initialized = False
        st.session_state.error = str(e)

with st.sidebar:
    st.title("⚙️ Configuration")
    
    api_key = st.text_input(
        "GROQ API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Enter your GROQ API key to enable LLM-powered responses"
    )
    
    if st.button("Initialize Chatbot", type="primary"):
        if api_key:
            try:
                st.session_state.chatbot = FundChatbot(
                    st.session_state.analyzer,
                    api_key=api_key
                )
                st.success("Chatbot initialized successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error initializing chatbot: {e}")
        else:
            st.warning("Please enter a GROQ API key")
    
    st.divider()
    
    st.subheader("📊 Data Overview")
    if st.session_state.chatbot_initialized:
        try:
            funds = st.session_state.analyzer.get_all_funds()
            st.write(f"**Total Funds:** {len(funds)}")
            st.write(f"**Funds:** {', '.join(funds)}")
            
            holdings_count = st.session_state.analyzer.get_holdings_count_by_fund()
            trades_count = st.session_state.analyzer.get_trades_count_by_fund()
            
            st.write(f"**Total Holdings:** {sum(holdings_count.values())}")
            st.write(f"**Total Trades:** {sum(trades_count.values())}")
        except Exception as e:
            st.error(f"Error loading data overview: {e}")
    else:
        st.error(f"Data not loaded: {st.session_state.get('error', 'Unknown error')}")

st.title("Fund Data Chatbot")
st.markdown("Ask questions about fund holdings, trades, and performance data.")

if not st.session_state.chatbot_initialized:
    st.error(f"Failed to load data: {st.session_state.get('error', 'Unknown error')}")
    st.info("Please ensure `holdings.csv` and `trades.csv` files are in the current directory.")
    st.stop()

if st.session_state.chatbot is None:
    st.info("Please initialize the chatbot using the sidebar to start chatting.")
    
    st.subheader("💡 Sample Questions")
    sample_questions = [
        "What are all the available funds?",
        "How many holdings does each fund have?",
        "Show me the performance of all funds",
        "What is the total PL_YTD for each fund?",
        "How many trades are there per fund?",
    ]
    
    for i, question in enumerate(sample_questions, 1):
        st.write(f"{i}. {question}")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask a question about the fund data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chatbot.answer(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    if st.session_state.messages:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

if st.session_state.chatbot_initialized:
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["📈 Performance", "💼 Holdings", "🔄 Trades"])
    
    with tab1:
        st.subheader("Fund Performance")
        try:
            performance_df = st.session_state.analyzer.get_fund_performance()
            st.dataframe(performance_df, use_container_width=True)
            
            if len(performance_df) > 0:
                st.bar_chart(performance_df.set_index("Fund")["Total_PL_YTD"])
        except Exception as e:
            st.error(f"Error loading performance data: {e}")
    
    with tab2:
        st.subheader("Holdings by Fund")
        try:
            holdings_count = st.session_state.analyzer.get_holdings_count_by_fund()
            holdings_df = pd.DataFrame(list(holdings_count.items()), columns=["Fund", "Holdings Count"])
            st.dataframe(holdings_df, use_container_width=True)
            
            if len(holdings_df) > 0:
                st.bar_chart(holdings_df.set_index("Fund"))
        except Exception as e:
            st.error(f"Error loading holdings data: {e}")
    
    with tab3:
        st.subheader("Trades by Fund")
        try:
            trades_count = st.session_state.analyzer.get_trades_count_by_fund()
            trades_df = pd.DataFrame(list(trades_count.items()), columns=["Fund", "Trades Count"])
            st.dataframe(trades_df, use_container_width=True)
            
            if len(trades_df) > 0:
                st.bar_chart(trades_df.set_index("Fund"))
        except Exception as e:
            st.error(f"Error loading trades data: {e}")
