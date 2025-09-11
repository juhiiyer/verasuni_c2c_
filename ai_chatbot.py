import random

# ------------------------------
# Bibble - AI Chatbot (Stage 1)
# ------------------------------

# Predefined sample questions (more can be added)
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

def bibble_chat():
    print("Hi, I’m Bibble, your personal AI ChatBot.")
    print("I'm going to assist you with any queries or tasks you have for me.\n")

    choice = input("Do you want sample questions you can ask me? (yes/no): ").strip().lower()

    if choice == "yes":
        # Pick 4 random questions without repetition
        examples = random.sample(sample_questions, 4)
        print("\nGreat! Here are some things you can try asking me:")
        for i, q in enumerate(examples, 1):
            print(f"{i}. {q}")
        print("\nNow, what would you like me to do?")
    else:
        print("\nAlright, no problem. What’s your query?")

    # Placeholder: user input for actual query
    user_query = input("You: ")

    # Placeholder: this is where ChatGPT API or DB lookup would happen
    print("\n[Bibble is thinking 🤔 ... connecting to AI/database]")
    print(f"(Pretend AI answer for now): I processed your query — '{user_query}'")

# Run the chatbot
if __name__ == "__main__":
    bibble_chat()
