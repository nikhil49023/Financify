Financify: 

Your AI-Powered Financial Co-Pilot 🚀
Financify is an intelligent personal finance application designed to make financial management simple, smart, and engaging for young Indians. It leverages the power of Generative AI to transform the tedious task of budget tracking into an effortless and insightful experience.

Links
🎥 Watch the Live Demo Video :- https://youtu.be/ZzutoRQfv2k

🌐 Interact with the Live Prototype

✨ About the Project
Traditional finance apps are often boring, complex, and easy to abandon. Financify tackles this problem head-on by creating a user-centric platform that is:

Effortless: Automatically extract transactions by simply uploading a bank statement screenshot.

Intelligent: Get personalized, proactive advice from an AI Financial Advisor that understands your unique spending habits.

Engaging: Learn about finance through bite-sized audio lessons and interactive quizzes.

This project was built for the OpenAI x NxtWave Buildathon, with a core mission to improve financial literacy in India, aligning with the RBI's national strategy for financial education.

🌟 Key Features
Financify is packed with features designed to provide a complete financial wellness toolkit:

🤖 AI Document Analyzer: Upload a bank statement image (PNG, JPG) and watch as the AI automatically extracts, categorizes, and populates your income and expenses.

💬 Conversational AI Advisor: Ask complex financial questions and receive personalized advice that analyzes your actual financial data to provide actionable steps.

📊 Interactive Dashboard: A clean, visual overview of your monthly income, expenses, savings, and progress towards your financial goals.

💡 Personalized Insights: Get AI-generated tips on crucial topics like tax-saving, insurance, investing, and the power of compounding.

🎧 'Fin-Bites' Audio Learning: An audiobook-style learning hub with bite-sized lessons on key financial concepts.

🧠 Integrated Quizzes: Test your knowledge after each audio lesson to reinforce learning and prepare for the upcoming Fin-IQ score.

👤 Secure Onboarding & Profile Management: A complete and secure user journey from login to profile creation.

🛠️ Tech Stack
This prototype was built using a modern, AI-centric Python stack, chosen for its ability to enable rapid development and seamless API integration.

Application Framework: Streamlit

Core Logic & Language: Python

Generative AI: Google Gemini API (gemini-1.5-flash) for its powerful multi-modal (text and vision) capabilities.

UI Components: streamlit-option-menu for clean navigation.

Data Handling: pandas

Data Visualization: plotly

Image Processing: Pillow

⚙️ Setup and Installation
To run this project locally, follow these steps:

1. Clone the repository:

git clone [https://github.com/nikhil49023/financify.git](https://github.com/nikhil49023/financify.git)
cd financify

2. Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Create and install dependencies:

First, create a file named requirements.txt in the root of your project folder and paste the following content into it:

streamlit
google-generativeai
pandas
streamlit-option-menu
Pillow
plotly
numpy

Now, run the following command in your terminal to install all the necessary libraries from this file:

pip install -r requirements.txt

4. Configure your API Key:

Create a .env file in the root directory.

Add your Gemini API key to this file:

GEMINI_API_KEY="YOUR_API_KEY_HERE"

Alternatively, the application will prompt you to enter the API key in the sidebar on the first run.

5. Run the Streamlit application:

python -m streamlit run app.py
🗺️ Future Roadmap
Financify is built with a vision for scale. Our future roadmap includes:

Implement the 'Fin-IQ Challenge': Launch the full gamification engine with scores, daily streaks, and competitive leaderboards.

'Goal Getter' Savings Goals: Allow users to set, track, and get advice on specific savings goals (e.g., "Trip to Goa").

Smart Tax Planner: A proactive tool to help users plan their tax-saving investments throughout the year.

Transition to a Production Stack: Evolve the application to a more scalable architecture using React (Next.js) for the frontend, Python (FastAPI) for the backend, and Firebase (Firestore) for the database.

Multi-Language Support: Expand the 'Fin-Bites' and AI Advisor to support multiple Indian regional languages.

👤 Author
Kilani Sai Nikhil

LinkedIn: www.linkedin.com/in/kilanisainikhil

GitHub: https://github.com/nikhil49023
