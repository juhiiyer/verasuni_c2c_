import requests
from bs4 import BeautifulSoup
import mysql.connector
from mysql.connector import Error
import time
import os
import praw
import json
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import ChromeOptions as Options

def create_mysql_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='root1234',
            database='verasuni'
        )
        if connection.is_connected():
            print("Connected to MySQL")
            return connection
    except Error as e:
        print("Error connecting to MySQL", e)
        return None

def create_tables(connection):
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS colleges (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255),
            website VARCHAR(255)
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS data_types (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100)
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS college_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            college_id INT,
            data_type_id INT,
            source_website VARCHAR(255),
            content TEXT,
            subthreads LONGTEXT,   --  store comments/answers as JSON
            FOREIGN KEY (college_id) REFERENCES colleges(id),
            FOREIGN KEY (data_type_id) REFERENCES data_types(id)
        );
    """)
    connection.commit()

def insert_college(connection, name, website):
    cursor = connection.cursor()
    cursor.execute("SELECT id FROM colleges WHERE name=%s", (name,))
    result = cursor.fetchone()
    if result:
        return result[0]
    cursor.execute("INSERT INTO colleges (name, website) VALUES (%s, %s)", (name, website))
    connection.commit()
    return cursor.lastrowid

def insert_data_type(connection, data_type_name):
    cursor = connection.cursor()
    cursor.execute("SELECT id FROM data_types WHERE name=%s", (data_type_name,))
    result = cursor.fetchone()
    if result:
        return result[0]
    cursor.execute("INSERT INTO data_types (name) VALUES (%s)", (data_type_name,))
    connection.commit()
    return cursor.lastrowid


def insert_college_data(connection, college_id, data_type_id, source_website, content, subthreads="[]"):
    cursor = connection.cursor()
    cursor.execute("""
        INSERT INTO college_data (college_id, data_type_id, source_website, content, subthreads)
        VALUES (%s, %s, %s, %s, %s)
    """, (college_id, data_type_id, source_website, content, subthreads))
    connection.commit()

def is_relevant_post(text):
    text = text.lower()

    whitelist_keywords = [
        "hostel", "mess", "food", "canteen", "faculty", "teacher", "professor", 
        "labs", "placement", "internship", "exam", "syllabus", "course", 
        "campus", "infrastructure", "review", "fees", "library",""
    ]
    blacklist_keywords = [
        "gf", "bf", "crush", "romance", "love", "relationship", "sex", 
        "dating", "breakup", "depressed", "suicidal", "marks", 
        "performance", "jealous", "girl", "boy", "marriage","home","mom","looking","selling"
    ]

    if any(word in text for word in blacklist_keywords):
        return False
    return any(word in text for word in whitelist_keywords)

def scrape_reddit_college_posts(subreddit_name, reddit_instance, connection):
    subreddit = reddit_instance.subreddit(subreddit_name)
    print(f"Scraping posts from Reddit subreddit: {subreddit_name}")
    data_type_id = insert_data_type(connection, "reddit_post")
    college_id = insert_college(connection, subreddit_name, f"https://www.reddit.com/r/{subreddit_name}")

    for post in subreddit.hot(limit=20):
        post_content = f"{post.title}\n{post.selftext}"
        
        post.comments.replace_more(limit=0)
        replies = [comment.body for comment in post.comments.list()]
        
        if is_relevant_post(post_content):
            insert_college_data(
                connection,
                college_id,
                data_type_id,
                f"https://www.reddit.com{post.permalink}",
                post_content,
                json.dumps(replies, ensure_ascii=False)  

        )

def scrape_shiksha_selenium(url, connection, chromedriver_path):
    if not os.path.exists(chromedriver_path):
        print(f"ChromeDriver not found at {chromedriver_path}. Please check the path.")
        return
    options = Options()
    options.add_argument("--headless")
    service = Service(chromedriver_path)
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    time.sleep(3)

    try:
        college_name_elem = driver.find_element(By.CSS_SELECTOR, 'h1.clg-name')
        college_name = college_name_elem.text.strip()
    except Exception:
        college_name = "Unknown Shiksha College"

    college_id = insert_college(connection, college_name, url)

    try:
        placements_elem = driver.find_element(By.ID, 'placements')
        placements_text = placements_elem.text.strip()
    except Exception:
        placements_text = "Placement data not found"

    data_type_id = insert_data_type(connection, 'placements')
    insert_college_data(connection, college_id, data_type_id, url, placements_text)
    driver.quit()

def scrape_careers360_college(url, connection):
    print(f"Scraping Careers360 URL: {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    college_name_tag = soup.find('h1')
    college_name = college_name_tag.text.strip() if college_name_tag else "Unknown Careers360 College"
    college_id = insert_college(connection, college_name, url)

    courses_tag = soup.find('div', {'id': 'courses'})
    courses_text = courses_tag.text.strip() if courses_tag else "Courses info not found"

    data_type_id = insert_data_type(connection, 'courses')
    insert_college_data(connection, college_id, data_type_id, url, courses_text)

def scrape_nirf_rankings(url, connection):
    print(f"Scraping NIRF URL: {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    tables = soup.find_all('table')

    content = ""
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            text = row.get_text(separator=" ").strip()
            if "Manipal" in text or "Vellore Institute of Technology" in text:
                content += text + "\n"

    if content:
        college_name = "NIRF Rankings (Manipal & VIT)"
        college_id = insert_college(connection, college_name, url)
        data_type_id = insert_data_type(connection, 'rankings')
        insert_college_data(connection, college_id, data_type_id, url, content)

def scrape_quora_college_spaces(url, connection):
    print(f"Scraping Quora URL: {url}")
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    college_name = url.split("/")[-1].replace("-", " ")
    college_id = insert_college(connection, college_name, url)
    data_type_id = insert_data_type(connection, "quora")

    question_elem = soup.find("h1")
    question_text = question_elem.get_text(strip=True) if question_elem else "Quora Topic"

    answers = [ans.get_text(strip=True) for ans in soup.find_all("div", {"class": "q-relative spacing_log_answer_content"})]

    insert_college_data(
        connection,
        college_id,
        data_type_id,
        url,
        question_text,
        json.dumps(answers, ensure_ascii=False)  
    )

def main():
    chromedriver_path = r'C:/Users/Shriya/Downloads/chromedriver-win64/chromedriver-win64/chromedriver.exe'
    connection = create_mysql_connection()
    if connection is None:
        print("Failed to connect to the database.")
        return

    create_tables(connection)

    reddit = praw.Reddit(
        client_id='OBeGZDYg0lRIVosN2zc37g',
        client_secret='Hj_3EnAf3Rzhl2FN-AYN6bC2w9I0Tg',
        user_agent='college_data_scraper_v1'
    )

    reddit_subreddits = ['manipal', 'Vit']  
    for subreddit in reddit_subreddits:
        try:
            scrape_reddit_college_posts(subreddit, reddit, connection)
            time.sleep(5)
        except Exception as e:
            print(f"Could not scrape subreddit {subreddit}: {e}")

    shiksha_urls = [
        'https://www.shiksha.com/university/manipal-university-udupi-50097',
        'https://www.shiksha.com/university/vellore-institute-of-technology-vit-vellore-51084'
    ]
    for url in shiksha_urls:
        scrape_shiksha_selenium(url, connection, chromedriver_path)
        time.sleep(3)

    careers360_urls = [
        'https://www.careers360.com/university/manipal-academy-of-higher-education-manipal',
        'https://www.careers360.com/university/vellore-institute-of-technology-vellore'
    ]
    for url in careers360_urls:
        scrape_careers360_college(url, connection)
        time.sleep(3)

    nirf_urls = [
        'https://www.nirfindia.org/2024/EngineeringRanking.html'
    ]
    for url in nirf_urls:
        scrape_nirf_rankings(url, connection)
        time.sleep(3)

    quora_urls = [
        'https://www.quora.com/topic/Manipal-University-India',
        'https://www.quora.com/topic/Vellore-Institute-of-Technology-VIT'
    ]
    for url in quora_urls:
        scrape_quora_college_spaces(url, connection)
        time.sleep(3)

    connection.close()
    print("All scraping completed.")

if __name__ == "__main__":
    main()
