import random
from collections import Counter
import os
import mysql.connector
import google.generativeai as genai

# --- API KEY CONFIGURATION ---
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(" GEMINI_API_KEY not set. Please set it as an environment variable.")

genai.configure(api_key=api_key)


# -----------------------------

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="dummy_data"
    )

def query_database(user_query):
    db = connect_db()
    cursor = db.cursor(dictionary=True)
    sql = "SELECT * FROM colleges WHERE name LIKE %s LIMIT 5"
    val = (f"%{user_query}%", )
    cursor.execute(sql, val)
    results = cursor.fetchall()
    cursor.close()
    db.close()
    return results

'''
def query_database(user_query):
    db = connect_db()
    cursor = db.cursor(dictionary=True)
    sql = "SELECT * FROM colleges WHERE name LIKE %s OR city LIKE %s LIMIT 5"
    val = (f"%{user_query}%", f"%{user_query}%")
    cursor.execute(sql, val)
    results = cursor.fetchall()
    cursor.close()
    db.close()
    return results
'''

def categorize_query(query):
    query_lower = query.lower()
    categories = {
        "placements": ["placement", "package", "job", "hiring", "recruitment"],
        "hostel": ["hostel", "dorm", "room", "accommodation"],
        "food": ["food", "mess", "canteen", "dining"],
        "faculty": ["teacher", "faculty", "professor", "lecturer"],
        "fees": ["fees", "cost", "tuition", "scholarship"],
        "ranking": ["rank", "nirf", "top", "best"],
        "alumni": ["alumni", "network", "seniors", "graduates"],
        "infrastructure": ["infrastructure", "library", "labs", "wifi", "sports"]
    }
    for category, keywords in categories.items():
        if any(word in query_lower for word in keywords):
            return category
    return "general"




# --- GEMINI RESPONSE FUNCTION ---
def get_gemini_response(user_query, db_results=None):
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"You are Bibble, a friendly AI assistant that helps students with college-related queries. Respond to the user's request. If there are database results, use them to inform your response. Here are some database results you can use: {db_results}. User query: {user_query}"

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Sorry, something went wrong: {e}"


# ----------------------------------

sample_questions = [
    "Can you generate a personalized comparison table of Hostel, Food, and Placements for College1 vs. College2?",
    "Make me a downloadable Excel sheet of the pros and cons of the food and hostel life there.",
    "Which college has the best student facilities and why?",
    "Make me a CSV file of colleges with their admission fees.",
    "Which college would be more suited for a social butterfly. ",
    "Which college would pay more attention to the academics and hostel life?",
    "Summarize campus life reviews of College1 into a short paragraph.",
    "which college has the better campus?",
    "which college's facilities are more suited for athletics?",
    "compare the college's infrastructures in detail. "
]








def bibble_chat(user_id="guest"):
    print("Hi, I’m Bibble, your personal AI ChatBot.")
    print("I'm going to assist you with any queries or tasks you have for me.\n")

    choice = input("Do you want sample questions you can ask me? (yes/no): ").strip().lower()
    if choice == "yes":
        examples = random.sample(sample_questions, 4)
        print("\nHere are some other things you can try asking me:")
        for i, q in enumerate(examples, 1):
            print(f"{i}. {q}")
    else:
        print("\nAlright, no problem. What’s your query?")

    while True:
        user_query = input("\nYou: ").strip()
        if user_query.lower() in {"exit", "quit"}:
            print("Bibble: Thanks for chatting! Goodbye 👋")
            break



        db_results = query_database(user_query)

        print("\n[Bibble is thinking 🤔 ...]")
        answer = get_gemini_response(user_query, db_results)
        print(f"Bibble: {answer}")




if __name__ == "__main__":
    print("Starting Bibble Chatbot...\n")
    bibble_chat()