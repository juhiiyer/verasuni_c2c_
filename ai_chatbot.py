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
        password="My gl@ss3s.",
        database="dummy_data"
    )


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


def log_user_interaction(user_id, query):
    category = categorize_query(query)
    db = connect_db()
    cursor = db.cursor()
    sql = "INSERT INTO user_logs (user_id, query, category) VALUES (%s, %s, %s)"
    cursor.execute(sql, (user_id, query, category))
    db.commit()
    cursor.close()
    db.close()


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
    "Can you generate a personalized comparison table of Hostel, Food, and Placements for XYZ College vs ABC College?",
    "Regenerate the table but sort the ratings by faculty quality first.",
    "Make me a downloadable Excel sheet of all Engineering colleges in Bangalore with their rankings.",
    "Summarize alumni reviews of Delhi University into a short paragraph.",
    "Compare the placement statistics of IIT Delhi and BITS Pilani.",
    "Which colleges in Mumbai have the best hostel facilities?",
    "Generate a chart of top 10 engineering colleges in India based on NIRF ranking.",
    "Give me a list of colleges in Chennai with good infrastructure and faculty reviews.",
    "Prepare a text summary of Computer Science programs across 3 colleges.",
    "Make me a CSV file of law colleges in Delhi with their admission fees."
]

user_logs_memory = []


def suggest_personalized_questions():
    if not user_logs_memory:
        return None
    words = " ".join(user_logs_memory).lower().split()
    keywords = [w for w in words if w not in {"the", "of", "in", "a", "to", "and"}]
    if not keywords:
        return None
    most_common = Counter(keywords).most_common(1)[0][0]
    related = [q for q in sample_questions if most_common in q.lower()]
    return random.sample(related, min(2, len(related))) if related else None


def analyze_user_interests():
    if not user_logs_memory:
        return None
    words = " ".join(user_logs_memory).lower().split()
    keywords = [w for w in words if w not in {"the", "of", "in", "a", "to", "and", "me", "my", "you"}]
    if not keywords:
        return None
    common = Counter(keywords).most_common(3)
    return [word for word, _ in common]


def bibble_chat(user_id="guest"):
    print("Hi, I’m Bibble, your personal AI ChatBot.")
    print("I'm going to assist you with any queries or tasks you have for me.\n")

    choice = input("Do you want sample questions you can ask me? (yes/no): ").strip().lower()
    if choice == "yes":
        personalized = suggest_personalized_questions()
        if personalized:
            print("\nBased on your interests, you might like:")
            for i, q in enumerate(personalized, 1):
                print(f"{i}. {q}")
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

        user_logs_memory.append(user_query)
        log_user_interaction(user_id, user_query)

        db_results = query_database(user_query)

        print("\n[Bibble is thinking 🤔 ...]")
        answer = get_gemini_response(user_query, db_results)
        print(f"Bibble: {answer}")

        if len(user_logs_memory) % 3 == 0:
            interests = analyze_user_interests()
            if interests:
                print(f"\n📊 Bibble Insight: I noticed you’ve been asking a lot about {', '.join(interests)}.")
                print("Would you like me to prepare a summary or comparison based on that?")


if __name__ == "__main__":
    print("Starting Bibble Chatbot...\n")
    bibble_chat()