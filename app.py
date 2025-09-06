import google.generativeai as genai
import os
import streamlit as st
import datetime
import pandas as pd
from streamlit_option_menu import option_menu
import json
from PIL import Image
import plotly.graph_objects as go
import numpy as np
from io import BytesIO

# --- 1. CONFIGURE API KEY & MODEL ---
# Use the vision-capable model for both text and image tasks
MODEL_NAME = 'gemini-1.5-flash'

def configure_api():
    """Configures the Gemini API, trying env vars, session state, or user input."""
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        gemini_api_key = st.session_state.get("gemini_api_key")

    if not gemini_api_key:
        with st.sidebar:
            st.header("API Key Required")
            st.warning("Gemini API Key not found. Please enter it below.")
            gemini_api_key = st.text_input("Enter your Gemini API Key", type="password", key="AIzaSyBwSBhOjhh2-CF47C2huFrl7JnXNU2Nvog")
            if gemini_api_key:
                st.session_state.gemini_api_key = gemini_api_key
                st.rerun()
            else:
                st.stop()

    try:
        genai.configure(api_key=gemini_api_key)
        return True
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {e}")
        with st.sidebar:
            st.error("Invalid API Key. Please check and re-enter.")
            if "gemini_api_key" in st.session_state:
                del st.session_state["gemini_api_key"]
        st.stop()

# --- 2. DEFINE THE CORE AI PROMPTS & KNOWLEDGE BASE ---
RBI_KNOWLEDGE_BASE = """
**RBI Financial Planning Guidelines:**
- The first and most important step in financial planning is to know your money well (income, expenses, assets, liabilities).
- The three key components of money management are Record Keeping, Budgeting, and Saving.
- A budget is a plan you commit to follow to prudently manage your money.
- Saving is putting aside money for future use; it should be done first and regularly.
- Common financial mistakes include spending frivolously, mindless use of credit cards, and investing based on half-knowledge.
- Good credit is a lifelong asset built by paying back debts on time and in full.
- The 'Rule of 72' is a simple trick to find out how long it will take to double your money.
- Investment decisions should be based on your risk profile, needs, and priorities, not on what a friend does.
- You should not invest money that is not yours and investable.
- An emergency fund should cover at least three months of living expenses.
- Insurance is a way to transfer risk and provide peace of mind in case of disasters.
"""

ADVISOR_SYSTEM_PROMPT = f"""
You are a conversational AI financial advisor named 'Financify,' which means 'Easy Wealth Companion.' Your persona is that of a friendly, knowledgeable, and trustworthy financial expert for young professionals in India. Your goal is to provide simple, actionable, and culturally relevant financial guidance. You are an expert in personal finance, and your advice is based on the following key guidelines:
{RBI_KNOWLEDGE_BASE}

Your tasks are:
1.  **Analyze User Data:** You will be provided with a user's monthly financial data in text format.
2.  **Provide a Budget Summary:** Give a clear, concise summary of their financial status, including total income, total expenses, and the remaining balance.
3.  **Give Actionable Advice:** Offer 2-3 specific, practical suggestions to improve their financial health. The advice must be based on the provided RBI guidelines.
4.  **Maintain a Trustworthy Tone:** The response must be polite, easy to read, and free of complex jargon. Always include a disclaimer at the end.
5.  **Roleplay:** If the user asks about a big purchase, respond as an advisor helping them analyze the decision based on their provided data, referencing your knowledge base.

**Disclaimer:** This advice is for educational purposes only and is not a substitute for professional financial consultation.
"""

INSIGHTS_PROMPT = f"""
As a conversational AI financial advisor for young professionals in India, generate 3 short, friendly, and actionable tips for a beginner on the following topics:
1. Tax-saving instruments (e.g., ELSS, PPF, NPS).
2. Different types of insurance (e.g., life, health, vehicle).
3. The importance of investing in mutual funds, stocks, and gold.
4. The concept of compounding, as explained in the provided text.

Base your tips on the following knowledge:
{RBI_KNOWLEDGE_BASE}

Keep the language simple and encouraging.
"""

DOCUMENT_ANALYSIS_PROMPT = """
You are an expert financial data extractor. Analyze the provided image of a bank statement, credit card statement, or transaction summary.
Your task is to identify the total monthly income and a list of all expenses.
Summarize the data into a clean JSON format.

- The 'income' should be a single numerical value representing the total credits or salary.
- The 'expenses' should be a list of JSON objects, where each object has a 'category' (e.g., "Rent", "Food", "Shopping", "UPI Transfer") and an 'amount' (a numerical value).
- If you cannot determine a specific category, use a general one like "Miscellaneous" or "Bank Transfer".
- Only extract debit/expense transactions for the expenses list. Ignore credit/income transactions in the expense list.
- If no income is found, set income to 0.

Example output:
{
  "income": 50000,
  "expenses": [
    {"category": "Rent", "amount": 15000},
    {"category": "Zomato", "amount": 1200},
    {"category": "Swiggy", "amount": 850},
    {"category": "UPI Transfer", "amount": 2000},
    {"category": "Shopping", "amount": 4500}
  ]
}

Provide only the JSON object in your response, with no other text or explanations.
"""

# --- 3. CREATE THE AI FUNCTIONS ---
def get_financial_advice(user_data_text, user_query):
    """Sends user data and the main prompt to the Gemini Pro API."""
    full_prompt = f"{ADVISOR_SYSTEM_PROMPT}\n\nUser's financial data:\n{user_data_text}\n\nUser's Question: {user_query}"
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

@st.cache_data
def get_insights_content():
    """Generates the content for the Insights page using a dedicated prompt."""
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(INSIGHTS_PROMPT)
        return response.text
    except Exception as e:
        return f"An error occurred while fetching insights: {e}"

def analyze_document_with_vision(uploaded_file):
    """Analyzes an uploaded document to extract income and expenses."""
    try:
        st.info("Analyzing document... This may take a moment.")
        model = genai.GenerativeModel(MODEL_NAME)

        if uploaded_file.type.startswith('image/'):
            image = Image.open(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload an image (PNG, JPG).")
            st.stop()

        response = model.generate_content([DOCUMENT_ANALYSIS_PROMPT, image])
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        data = json.loads(cleaned_response)

        income = data.get("income", 0.0)
        expenses_list = data.get("expenses", [])

        st.session_state.income = float(income)
        st.session_state.income_text = str(income)

        expense_entries = [{'category': item['category'], 'amount': float(item['amount'])} for item in expenses_list if item.get('category') and float(item.get('amount', 0)) > 0]
        st.session_state.expense_entries = expense_entries

        st.success("Analysis complete! Your financial data has been populated in the 'Manual Entry' tab. Please review and save.")
        st.rerun()

    except json.JSONDecodeError:
        st.error("AI analysis failed. The model did not return valid data. Please try a clearer document.")
    except Exception as e:
        st.error(f"An error occurred during document analysis: {e}")

# --- 4. DEFINE PAGE FUNCTIONS ---
def get_time_based_greeting():
    """Returns a greeting based on the time of day."""
    current_hour = datetime.datetime.now().hour
    if 5 <= current_hour < 12:
        return "Good morning"
    elif 12 <= current_hour < 18:
        return "Good afternoon"
    else:
        return "Good evening"

def page_login():
    """Login Page with a professional and user-friendly design."""
    st.title("Welcome to Financify")
    st.markdown("Your personal guide to financial wellness. Simple, smart, and secure.")
    st.markdown("---")

    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.markdown("### Take Control of Your Finances")
        st.markdown("""
        - **AI-Powered Advice:** Get personalized financial tips based on RBI guidelines.
        - **Budget Tracking:** Easily monitor your income and expenses.
        - **Insightful Analytics:** Visualize your spending habits with charts and graphs.
        - **Secure & Private:** Your data is always protected.
        """)

    with col2:
        with st.container(border=True):
            st.subheader("Login to Your Account")
            with st.form("login_form"):
                email_or_phone = st.text_input("Email or Phone Number", placeholder="you@example.com", key="login_email")
                password = st.text_input("Password", type="password", placeholder="••••••••", key="login_password")
                submitted = st.form_submit_button("Login", use_container_width=True)

                if submitted:
                    if email_or_phone and password:
                        st.session_state.is_logged_in = True
                        st.session_state.onboarded = False
                        st.rerun()
                    else:
                        st.error("Please enter both email/phone and password.")

def page_onboarding_form():
    """Onboarding form that appears after login."""
    st.title("Create Your Profile")
    st.markdown("Let's get to know you better to personalize your financial journey.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.image("poster.png", caption="Empower Your Financial Future with Financify", use_container_width=True)

    with col2:
        with st.form("onboarding_form"):
            with st.container(border=True):
                form_col1, form_col2 = st.columns(2)
                with form_col1:
                    name = st.text_input("Full Name *", placeholder="e.g., nikhil", help="Your name helps us personalize your experience.")
                with form_col2:
                    age = st.number_input("Age", min_value=18, max_value=100, value=25, help="Your age helps in suggesting age-appropriate financial products.")

                occupation = st.text_input("Occupation *", placeholder="e.g., Software Engineer, Doctor", help="Knowing your profession can help us tailor advice.")
                sector = st.selectbox("Sector", ["Private", "Government", "Farming", "Defense", "Business", "Student", "Other"], help="Select the sector you work in.")

                st.markdown("---")
                management_type = st.radio("How will you be using this app?", ["For myself (Individual)", "For my family (Family)"], horizontal=True)
                is_family_management = ("Family" in management_type)

                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button("🚀 Complete Profile & Get Started", use_container_width=True, type="primary")

                if submitted:
                    if name and occupation:
                        st.session_state.name = name
                        st.session_state.age = age
                        st.session_state.occupation = occupation
                        st.session_state.sector = sector
                        st.session_state.is_family_management = is_family_management
                        st.session_state.onboarded = True
                        st.success("Profile created successfully! Redirecting to your dashboard...")
                        st.rerun()
                    else:
                        st.warning("Please fill out all required fields marked with *.")

def page_home():
    """The new Home/Dashboard Page with enhanced visuals and actionable insights."""
    st.markdown(f"<h3 style='color: #d1d0cd;'>{get_time_based_greeting()}, {st.session_state.name.capitalize()}!</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #d1d0cd;'>Here's your financial overview at a glance.</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    <style>
        .metric-card { background-color: #363131; border-radius: 10px; padding: 20px; text-align: center; border: 1px solid #e6e6e6; transition: transform 0.2s; color: #d1d0cd; }
        .metric-card:hover { transform: translateY(-5px); }
        .progress-container { position: relative; height: 15px; background-color: #474646; border-radius: 7px; overflow: hidden; margin-top: 10px; }
        .progress-marker { position: absolute; top: 0; bottom: 0; width: 4px; background-color: #FFFFFF; transform: translateX(-2px); }
    </style>
    """, unsafe_allow_html=True)

    if 'income' not in st.session_state or 'expense_entries' not in st.session_state or not st.session_state.get('income'):
        st.info("Welcome! Please add your income and expenses to activate your financial dashboard.")
        if st.button("➕ Add Financials", use_container_width=True):
            st.session_state.page = "Add"
            st.rerun()
        return

    try:
        income = float(st.session_state.get('income', 0.0))
        expenses_data = {entry['category']: entry['amount'] for entry in st.session_state.get('expense_entries', [])}
        total_expenses = sum(expenses_data.values())
        balance = income - total_expenses

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='metric-card'><div style='color: #21ba45; font-weight: bold;'>INCOME</div><h2 style='color: white; margin: 5px 0;'>₹{income:,.0f}</h2><small style='color:white;'>Monthly</small></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><div style='color: #db2828; font-weight: bold;'>EXPENSES</div><h2 style='color: white; margin: 5px 0;'>₹{total_expenses:,.0f}</h2><small style='color:white;'>This Month</small></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><div style='color: #6a6aff; font-weight: bold;'>SAVINGS</div><h2 style='color: white; margin: 5px 0;'>₹{balance:,.0f}</h2><small style='color: white;'>Remaining Balance</small></div>", unsafe_allow_html=True)

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                savings_pct = int((balance / income) * 100) if income > 0 else 0
                savings_pct = max(0, savings_pct)
                recommended_savings_pct = 20
                bar_color = "#21ba45" if savings_pct >= recommended_savings_pct else "#db2828"

                st.markdown("<h5 style='color: white; text-align: center;'>Monthly Savings Rate</h5>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='color: {bar_color}; text-align: center; margin-bottom: 0;'>{savings_pct}%</h1>", unsafe_allow_html=True)
                st.markdown(f"<div class='progress-container'><div style='width: {savings_pct}%; background-color: {bar_color}; height: 100%;'></div><div class='progress-marker' style='left: {recommended_savings_pct}%;' title='Recommended: {recommended_savings_pct}%'></div></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size: 12px; color: white; text-align: center; margin-top: 5px;'>Goal: {recommended_savings_pct}% (Marked by 🏳️)</div>", unsafe_allow_html=True)

        with col2:
            with st.container(border=True):
                target_income = income * 1.20
                progress_to_target = int((income / target_income) * 100) if target_income > 0 else 100
                st.markdown("<h5 style='color: white; text-align: center;'>Target Monthly Income</h5>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='color: #6a6aff; text-align: center; margin-bottom: 0;'>₹{target_income:,.0f}</h1>", unsafe_allow_html=True)
                st.markdown(f"<div class='progress-container'><div style='width: {progress_to_target}%; background-color: #6a6aff; height: 100%;'></div></div>", unsafe_allow_html=True)
                st.markdown("<div style='font-size: 12px; color: #d1d0cd; text-align: center; margin-top: 5px;'>A goal for financial growth</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.subheader("Personalized Suggestions")
            sug_col1, sug_col2 = st.columns(2)
            with sug_col1:
                st.markdown("##### 💡 Maximize Your Savings")
                st.markdown("- **Review Subscriptions:** Cancel any unused services (e.g., streaming, apps).")
                st.markdown("- **Automate Savings:** Set up an automatic transfer to a savings account on payday.")
                if savings_pct < recommended_savings_pct:
                    st.warning("**Action Needed:** Your savings are below the recommended 20%. Try the '50/30/20 rule' - 50% for needs, 30% for wants, and 20% for savings.")
                else:
                    st.success("**Great Job!** You're meeting your savings goal. Consider investing the extra amount for higher returns.")
            with sug_col2:
                st.markdown("##### 🚀 Boost Your Income")
                st.markdown("- **Freelance Your Skills:** Offer your professional skills on platforms like Upwork or Fiverr.")
                st.markdown("- **Invest for Passive Income:** Explore dividend-paying stocks or mutual funds.")
                st.markdown("- **Skill Development:** Invest in a course to increase your value in the job market for a higher salary.")

    except Exception as e:
        st.error(f"Error displaying dashboard: {e}")
        st.warning("Please check your income and expense data on the 'Add' page.")

def page_advisor():
    """The AI Advisor Chat page."""
    st.markdown("### 🤖 AI Financial Advisor")
    st.markdown("<p style='color: #d1d0cd;'>Your personal guide to smarter financial decisions.</p>", unsafe_allow_html=True)

    if 'income' not in st.session_state or not st.session_state.get('income'):
        st.info("Please add your financial details on the 'Add' page first to use the AI advisor.")
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": f"Hello {st.session_state.name.capitalize()}! I'm Financify. How can I help you with your finances today?"}]
    if "processing_response" not in st.session_state:
        st.session_state.processing_response = False

    with st.container(height=500, border=True):
        for message in st.session_state.chat_history:
            avatar = "🤖" if message["role"] == "assistant" else "🧑‍💻"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

    user_query = st.chat_input("Ask me anything about your finances...")

    if user_query and not st.session_state.processing_response:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.processing_response = True
        st.rerun()

    if st.session_state.processing_response:
        last_user_query = st.session_state.chat_history[-1]["content"]
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                expenses_text = ", ".join([f"{e['category']}: {e['amount']}" for e in st.session_state.get('expense_entries', [])])
                user_data = f"Income: {st.session_state.get('income', 0)}\nExpenses: {expenses_text}"
                advice = get_financial_advice(user_data, last_user_query)
                st.session_state.chat_history.append({"role": "assistant", "content": advice})
                st.session_state.processing_response = False
                st.rerun()

def page_insights():
    """The Insights page with a visual deep-dive into financial health."""
    st.header("Financial Insights")
    st.markdown("A visual deep-dive into your spending and financial health.")
    st.markdown("---")

    st.markdown("""
    <style>
        .insight-card { background-color: #2a2a2b; border-radius: 10px; padding: 25px; border: 1px solid #444; height: 100%; }
        .progress-bar-custom { height: 12px; background-color: #444; border-radius: 6px; overflow: hidden; margin-top: 10px; }
        .progress-bar-inner { height: 100%; background: linear-gradient(to right, #6a6aff, #86d472); border-radius: 6px; }
    </style>
    """, unsafe_allow_html=True)

    if not st.session_state.get('income') or not st.session_state.get('expense_entries'):
        st.info("Please add your income and expenses on the 'Add' page to unlock your financial insights.")
        return

    try:
        income = float(st.session_state.income)
        expenses_data = {entry['category']: entry['amount'] for entry in st.session_state.expense_entries}
        total_expenses = sum(expenses_data.values())
        balance = income - total_expenses
        savings_rate = (balance / income * 100) if income > 0 else 0

        col1, col2 = st.columns(2)
        with col1:
            # Dynamic Risk Profile Card
            if savings_rate < 10:
                risk_profile, color, position, description = "Aggressive", "#db2828", "85%", "High risk tolerance, seeking high growth."
            elif 10 <= savings_rate < 25:
                risk_profile, color, position, description = "Balanced", "#FFD700", "50%", "Moderate risk with steady returns."
            else:
                risk_profile, color, position, description = "Conservative", "#21ba45", "15%", "Low risk tolerance, prioritizing capital preservation."

            st.markdown(f"""
            <div class="insight-card">
                <h5 style='color: #d1d0cd; margin-bottom: 15px;'>Your Simulated Risk Profile</h5>
                <div style='text-align: center;'>
                    <h2 style='color: {color}; margin: 0;'>{risk_profile.upper()}</h2>
                    <p style='color: #aaa; font-size: 14px;'>{description}</p>
                </div>
                <div style='position: relative; height: 8px; background: #444; border-radius: 4px; margin-top: 20px;'>
                    <div style='position: absolute; left: {position}; top: -6px; width: 20px; height: 20px; background: {color}; border-radius: 50%; transform: translateX(-50%); border: 3px solid #2a2a2b;'></div>
                </div>
                <div style='display:flex; justify-content: space-between; color: #888; font-size: 12px; margin-top: 10px;'>
                    <span>Conservative</span><span>Balanced</span><span>Aggressive</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Monthly Budget Card
            budget_goal = income * 0.8
            spent_percentage = (total_expenses / budget_goal * 100) if budget_goal > 0 else 0
            remaining_budget = budget_goal - total_expenses
            status_color = "#21ba45" if remaining_budget >= 0 else "#db2828"
            status_text = "On Track" if remaining_budget >= 0 else "Over Budget"

            st.markdown(f"""
            <div class="insight-card">
                <h5 style='color: #d1d0cd; margin-bottom: 15px;'>Monthly Budget Status</h5>
                <div style='display: flex; justify-content: space-between; align-items: center;'><span style='color: #aaa; font-size: 14px;'>Spent</span><span style='color: {status_color}; font-weight: bold;'>{status_text}</span></div>
                <div style='display: flex; justify-content: space-between; align-items: baseline;'><h3 style='color: white; margin: 0;'>₹{total_expenses:,.0f}</h3><span style='color: #aaa; font-size: 14px;'>of ₹{budget_goal:,.0f}</span></div>
                <div class='progress-bar-custom'><div class='progress-bar-inner' style='width: {min(spent_percentage, 100)}%;'></div></div>
                <p style='color: #aaa; margin-top: 10px; font-size: 14px;'>{'You have ₹{:,.0f} left.'.format(remaining_budget) if remaining_budget >= 0 else 'You are ₹{:,.0f} over budget.'.format(abs(remaining_budget))}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Expense Breakdown")
        with st.container(border=True):
            if expenses_data:
                # Sort expenses from largest to smallest to ensure consistency
                sorted_expenses = sorted(expenses_data.items(), key=lambda item: item[1], reverse=True)
                labels = [item[0] for item in sorted_expenses]
                values = [item[1] for item in sorted_expenses]
                colors = ['#6a6aff', '#FF6347', '#32CD32', '#FFD700', '#4682B4', '#9370DB']

                col1, col2 = st.columns([0.5, 0.5])
                with col1:
                    # Create the pie chart with sorted data
                    fig = go.Figure(data=[go.Pie(
                        labels=labels,
                        values=values,
                        hole=.6,
                        marker=dict(colors=colors), # Use the defined colors
                        hoverinfo='label+percent',
                        textinfo='value',
                        texttemplate='₹%{value:,.0f}',
                        textfont_size=14,
                        sort=False # Prevent plotly from re-sorting, so colors match the legend
                    )])
                    fig.update_layout(
                        showlegend=False,
                        height=400,
                        margin=dict(t=0, b=0, l=0, r=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#d1d0cd')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    # Create the text breakdown (legend) using the same sorted data and colors
                    st.markdown("<h5 style='color: #d1d0cd; padding-top: 20px;'>Top Spending Categories</h5>", unsafe_allow_html=True)
                    for i, (category, amount) in enumerate(sorted_expenses):
                        percentage = (amount / total_expenses) * 100 if total_expenses > 0 else 0
                        color = colors[i % len(colors)] # Cycle through colors
                        st.markdown(f"<div style='display: flex; align-items: center; margin-bottom: 12px;'><div style='width: 15px; height: 15px; background-color: {color}; border-radius: 3px; margin-right: 10px;'></div><div style='flex-grow: 1; color: #d1d0cd;'>{category}</div><div style='font-weight: bold; color: white;'>₹{amount:,.0f}</div><div style='width: 50px; text-align: right; color: #aaa;'>{percentage:.0f}%</div></div>", unsafe_allow_html=True)
            else:
                st.info("No expenses logged to display a chart.")

        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.subheader("💡 AI-Generated Financial Tips")
            with st.spinner("Generating personalized tips based on RBI guidelines..."):
                tips_content = get_insights_content()
                tips_content = tips_content.replace("###", "#####").replace("**", "")
                st.markdown(tips_content)

    except Exception as e:
        st.error(f"An error occurred while generating insights: {e}")

def page_add():
    """The 'Add' Page for income and expenses with an improved UI."""
    st.header("Add Transactions")
    st.markdown("Log your income and expenses to keep your financial dashboard up-to-date.")
    st.markdown("---")

    if 'expense_entries' not in st.session_state:
        st.session_state.expense_entries = [{'category': '', 'amount': 0.0}]

    tab1, tab2 = st.tabs(["✍️ Manual Entry", "📄 Upload Statement"])

    with tab1:
        with st.form("transaction_form"):
            st.markdown("##### Your Monthly Income")
            income_input = st.number_input("Enter your total monthly income (₹)", min_value=0.0, value=float(st.session_state.get('income', 0.0)), step=1000.0, format="%.2f")

            st.markdown("---")
            st.markdown("##### Your Monthly Expenses")

            # Create a container for the dynamic list of expenses
            expense_container = st.container()
            with expense_container:
                for i, entry in enumerate(st.session_state.expense_entries):
                    col1, col2 = st.columns([4, 3])
                    entry['category'] = col1.text_input("Expense Category", value=entry['category'], key=f"cat_{i}", placeholder="e.g., Rent")
                    entry['amount'] = col2.number_input("Amount (₹)", min_value=0.0, value=float(entry['amount']), key=f"amt_{i}", step=100.0, format="%.2f")

            # Form submission button
            submitted = st.form_submit_button("💾 Save All Transactions", type="primary", use_container_width=True)

            if submitted:
                st.session_state.income = income_input
                # Filter out empty entries before saving
                valid_entries = [e for e in st.session_state.expense_entries if e['category'].strip() and e['amount'] > 0]
                if not valid_entries:
                    st.warning("Please add at least one valid expense.")
                else:
                    st.session_state.expense_entries = valid_entries
                    st.success("Financial data updated successfully!")
                    st.rerun()

        # Button to add a new expense row (outside the form)
        if st.button("➕ Add Another Expense", use_container_width=True):
            st.session_state.expense_entries.append({'category': '', 'amount': 0.0})
            st.rerun()

    with tab2:
        st.subheader("Automated Statement Analysis")
        st.info("Upload an image of your bank statement or transaction summary to automatically extract your income and expenses.")
        with st.container(border=True):
            uploaded_file = st.file_uploader("Upload a statement image (PNG, JPG, etc.)", type=['png', 'jpg', 'jpeg', 'webp'])
            if uploaded_file:
                if st.button("🤖 Analyze with AI", use_container_width=True, type="primary"):
                    analyze_document_with_vision(uploaded_file)

def page_fin_bites():
    """The Fin-Bites page with a professional, music-player-like UI and an integrated quiz."""
    st.header("🎧 Fin-Bites")
    st.markdown("Bite-sized audio lessons to boost your financial literacy on the go.")
    st.markdown("---")

    # --- Custom CSS for the Spotify-like UI ---
    st.markdown("""
    <style>
        /* Main player card */
        .player-card {
            background-color: #121212; /* Darker charcoal */
            border: 1px solid #282828;
            border-radius: 12px;
            padding: 24px;
        }
        .player-card img {
            border-radius: 8px;
        }
        /* Playlist item styling */
        .playlist-item {
            padding: 12px 15px;
            margin-bottom: 8px;
            border-radius: 8px;
            background-color: #282828;
            cursor: pointer;
            transition: background-color 0.2s;
            border: 1px solid #282828;
        }
        .playlist-item:hover {
            background-color: #3a3a3a;
        }
        .playlist-item-selected {
            background-color: #1DB954; /* Spotify Green */
            border: 1px solid #1DB954;
            color: white;
        }
        .playlist-item-selected:hover {
            background-color: #1ED760;
        }
        /* Remove default button styling for playlist */
        div[data-testid="stButton"] > button.playlist-button {
            background-color: transparent;
            color: #d1d0cd;
            text-align: left;
            padding: 0;
            border: none;
            width: 100%;
        }
        div[data-testid="stButton"] > button.playlist-button:hover {
            color: white;
        }
        /* Action bar icon buttons */
        .action-bar {
            display: flex;
            justify-content: space-around;
            align-items: center;
            margin-top: 20px;
            padding-top: 15px;
            border-top: 1px solid #282828;
        }
        .action-button {
            background: none;
            border: none;
            color: #b3b3b3;
            font-size: 24px;
            cursor: pointer;
            transition: color 0.2s;
        }
        .action-button:hover {
            color: white;
        }
        .quiz-button {
            color: #1DB954; /* Emerald Green */
            font-weight: bold;
        }
        .quiz-button:hover {
            color: #1ED760;
        }
        /* Quiz dialog styling */
        div[data-testid="stDialog"] > div:first-child {
            background-color: #282828;
            border-radius: 12px;
        }
    </style>
    """, unsafe_allow_html=True)

    # --- Data for Fin-Bites lessons, audio, thumbnails, and quizzes ---
    fin_bites_lessons = {
        "Chapter 1: Needs vs Wants": {
            "creator": "Sourced from RBI Publications",
            "image_url": "thumb.png",
            "audio_url": "chap.mp3", # Placeholder
            "quiz": [
                {"question": "What does compounding primarily rely on?", "options": ["Initial investment only", "Reinvesting earnings", "Market volatility", "Frequent withdrawals"], "answer": "Reinvesting earnings"},
                {"question": "The 'Rule of 72' helps estimate...", "options": ["Monthly budget", "Tax liability", "Time to double money", "Credit score"], "answer": "Time to double money"},
            ]
        },
        "Understanding Mutual Funds": {
            "creator": "Sourced from RBI Publications",
            "image_url": "https://images.unsplash.com/photo-1624953587687-e271b715985e?q=80&w=870&auto=format&fit=crop",
            "audio_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3", # Placeholder
            "quiz": [
                {"question": "What does 'diversification' in a mutual fund mean?", "options": ["Investing in only one stock", "Spreading investments across various assets", "Only investing in gold", "Avoiding the stock market"], "answer": "Spreading investments across various assets"},
                {"question": "What is a 'Systematic Investment Plan' (SIP)?", "options": ["A one-time lump sum investment", "A plan to sell all stocks", "Investing a fixed amount regularly", "A type of insurance"], "answer": "Investing a fixed amount regularly"},
            ]
        },
        "Basics of Tax Saving": {
            "creator": "Sourced from RBI Publications",
            "image_url": "https://images.unsplash.com/photo-1554224155-1696413565d3?q=80&w=870&auto=format&fit=crop",
            "audio_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3", # Placeholder
            "quiz": [
                {"question": "Which is a popular tax-saving instrument under Section 80C in India?", "options": ["Savings Account", "Public Provident Fund (PPF)", "Credit Card points", "A foreign currency account"], "answer": "Public Provident Fund (PPF)"},
                {"question": "ELSS funds have a lock-in period of how many years?", "options": ["1 year", "3 years", "5 years", "10 years"], "answer": "3 years"},
            ]
        }
    }

    # --- Initialize Session State ---
    if 'selected_lesson' not in st.session_state:
        st.session_state.selected_lesson = next(iter(fin_bites_lessons))
    if 'show_quiz' not in st.session_state:
        st.session_state.show_quiz = False
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}

    # --- UI Layout ---
    playlist_col, player_col = st.columns([1, 1.5], gap="large")

    # --- Playlist Column ---
    with playlist_col:
        st.subheader("Playlist")
        for i, lesson_title in enumerate(fin_bites_lessons.keys()):
            is_selected = (st.session_state.selected_lesson == lesson_title)
            item_class = "playlist-item-selected" if is_selected else "playlist-item"
            
            st.markdown(f'<div class="{item_class}">', unsafe_allow_html=True)
            if st.button(f"**{lesson_title}**\n\n<small>{fin_bites_lessons[lesson_title]['creator']}</small>", key=f"lesson_{i}", use_container_width=True):
                st.session_state.selected_lesson = lesson_title
                st.session_state.show_quiz = False
                st.session_state.current_question = 0
                st.session_state.user_answers = {}
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # --- Player and Quiz Column ---
    with player_col:
        lesson_data = fin_bites_lessons[st.session_state.selected_lesson]
        quiz_questions = lesson_data["quiz"]

        with st.container(border=False):
            st.markdown('<div class="player-card">', unsafe_allow_html=True)
            
            # Player content: Cover Art on left, Info on right
            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(lesson_data["image_url"])
            with c2:
                st.subheader(st.session_state.selected_lesson)
                st.caption(f"{lesson_data['creator']}")
                st.audio(lesson_data["audio_url"], format='audio/mp3')

            # Action Bar
            st.markdown('<div class="action-bar">', unsafe_allow_html=True)
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                if st.button("?", key="quiz_button", help="Take the Quiz", use_container_width=True):
                    st.session_state.show_quiz = True
                    st.session_state.current_question = 0
                    st.session_state.user_answers = {}
                    st.rerun()
            with b2:
                st.button("♡", key="like_button", help="Favorite", use_container_width=True)
            with b3:
                st.button("🔗", key="share_button", help="Share", use_container_width=True)
            with b4:
                st.button("⚑", key="report_button", help="Report Issue", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    # --- Quiz Modal Logic ---
    if st.session_state.show_quiz:
        with st.dialog("Fin-Bite Quiz", width="large"):
            st.subheader(f"Quiz: {st.session_state.selected_lesson}")
            st.markdown("---")

            if st.session_state.current_question >= len(quiz_questions):
                score = sum(1 for i, q in enumerate(quiz_questions) if st.session_state.user_answers.get(i) == q["answer"])
                st.success(f"Quiz Complete! Your Score: {score} / {len(quiz_questions)}")
                st.balloons()
                if st.button("Close", use_container_width=True):
                    st.session_state.show_quiz = False
                    st.rerun()
                return

            q_idx = st.session_state.current_question
            question_data = quiz_questions[q_idx]
            
            st.markdown(f"**Question {q_idx + 1}/{len(quiz_questions)}**")
            user_choice = st.radio(question_data["question"], question_data["options"], key=f"q_{q_idx}", index=None)

            st.markdown("---")
            
            nav_cols = st.columns([1, 1, 2])
            with nav_cols[0]:
                if st.button("⬅️ Previous", use_container_width=True, disabled=(q_idx == 0)):
                    st.session_state.current_question -= 1
                    st.rerun()
            with nav_cols[2]:
                if st.button("Submit & Next ➡️", use_container_width=True, type="primary", disabled=(user_choice is None)):
                    st.session_state.user_answers[q_idx] = user_choice
                    st.session_state.current_question += 1
                    
                    if user_choice == question_data["answer"]:
                        st.toast("Correct! 🎉", icon="✅")
                    else:
                        st.toast(f"Not quite! The answer was: {question_data['answer']}", icon="❌")
                    st.rerun()

def page_profile():
    """A professional and attractive user profile page."""
    st.header("My Profile")
    st.markdown("Manage your account settings and personal information.")
    st.markdown("---")

    st.markdown("""
    <style>
        .profile-info-row {
            display: flex;
            align-items: center;
            font-size: 16px;
            margin-bottom: 15px;
            color: #d1d0cd;
        }
        .profile-info-row span:first-child {
            font-weight: bold;
            color: #aaa;
            width: 180px; /* Fixed width for labels */
        }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.image("https://static.vecteezy.com/system/resources/thumbnails/009/292/244/small/default-avatar-icon-of-social-media-user-vector.jpg", width=150)
        st.markdown(f"<h3 style='text-align: left; margin-top: 10px;'>{st.session_state.name.capitalize()}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: left; color: #aaa; margin-top: -10px;'>{st.session_state.occupation.capitalize()}</p>", unsafe_allow_html=True)

        if st.button("🔒 Logout", use_container_width=True, type="primary"):
            for key in list(st.session_state.keys()):
                if key not in ['is_logged_in', 'onboarded']:
                    del st.session_state[key]
            st.session_state.is_logged_in = False
            st.session_state.onboarded = False
            st.success("You have been logged out.")
            st.rerun()

    with col2:
        with st.container(border=True):
            st.subheader("Account Information")
            st.markdown(f"""
                <div class="profile-info-row"><span>👤 Full Name:</span><span>{st.session_state.name.capitalize()}</span></div>
                <div class="profile-info-row"><span>🎂 Age:</span><span>{st.session_state.age}</span></div>
                <div class="profile-info-row"><span>🏢 Sector:</span><span>{st.session_state.sector}</span></div>
                <div class="profile-info-row"><span>💼 Management Type:</span><span>{'Family' if st.session_state.is_family_management else 'Individual'}</span></div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.container(border=True):
            st.subheader("Preferences")
            st.toggle("Enable Email Notifications", value=True)
            st.selectbox("Theme", ["Dark", "Light"], help="Theme switching is a visual feature only.")
            st.selectbox("Language", ["English (India)"])

# --- 6. MAIN APPLICATION LOGIC ---
st.set_page_config(page_title="Financify", layout="wide")

# Initialize session state variables
default_session_state = {
    'is_logged_in': False, 'onboarded': False, 'name': "User",
    'income': 0.0, 'expense_entries': [], 'page': 'Home'
}
for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Main application router
if not st.session_state.is_logged_in:
    page_login()
elif not st.session_state.onboarded:
    page_onboarding_form()
else:
    # Configure API only if needed by the current page
    if st.session_state.get('page') in ["Advisor", "Insights", "Add"]:
        if not configure_api():
            st.stop() # Stop if API key is invalid or not provided

    with st.container():
        page = option_menu(
            menu_title=None,
            options=["Home", "Advisor", "Add", "Insights", "Fin-Bites", "Profile"],
            icons=["house-door-fill", "robot", "plus-circle-fill", "bar-chart-line-fill", "sound-wave", "person-circle"],
            menu_icon="cast",
            default_index=["Home", "Advisor", "Add", "Insights", "Fin-Bites", "Profile"].index(st.session_state.page),
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#212121"},
                "icon": {"color": "#fafafa", "font-size": "22px"},
                "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#424242"},
                "nav-link-selected": {"background-color": "#424242", "color": "#fafafa"},
            }
        )
    st.session_state.page = page

    page_functions = {
        "Home": page_home, "Advisor": page_advisor, "Add": page_add,
        "Insights": page_insights, "Fin-Bites": page_fin_bites, "Profile": page_profile
    }

    # Call the correct page function based on selection
    if page in page_functions:
        page_functions[page]()
