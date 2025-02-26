import json
import sqlite3
from typing import Optional, Iterator
import streamlit as st
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import bcrypt
import requests
from phi.agent import Agent
from phi.model.google import Gemini
from phi.workflow import Workflow, RunResponse, RunEvent
from phi.storage.workflow.sqlite import SqlWorkflowStorage
from phi.tools.duckduckgo import DuckDuckGo
from phi.utils.log import logger

load_dotenv()
ga = os.getenv("GOOGLE_API_KEY")
n_api = os.getenv("NEWS_API")

class NewsArticle(BaseModel):
    title: str = Field(..., description="Title of the article.")
    url: str = Field(..., description="Link to the article.")
    summary: Optional[str] = Field(None, description="Summary of the article if available.")

class SearchResults(BaseModel):
    articles: list[NewsArticle]

class BlogPostGenerator(Workflow):
    searcher: Agent = Agent(
        model=Gemini(id="gemini-2.0-flash", api_key=ga),
        tools=[DuckDuckGo()],
        instructions=["Given a topic, search for the top 5 articles."],
        response_model=SearchResults,
    )

    writer: Agent = Agent(
        model=Gemini(id="gemini-2.0-flash", api_key=ga),
        instructions=[
            "You will be provided with a topic and a list of top articles on that topic.",
            "Carefully read each article and generate a New York Times worthy blog post on that topic.",
            "Break the blog post into sections and provide key takeaways at the end.",
            "Make sure the title is catchy and engaging.",
            "Always provide sources, do not make up information or sources.",
        ],
        markdown=True,
    )

    def run(self, topic: str, use_cache: bool = True) -> Iterator[RunResponse]:
        logger.info(f"Generating a blog post on: {topic}")

        if use_cache:
            cached_blog_post = self.get_cached_blog_post(topic)
            if cached_blog_post:
                yield RunResponse(content=cached_blog_post, event=RunEvent.workflow_completed)
                return

        search_results: Optional[SearchResults] = self.get_search_results(topic)
        if search_results is None or len(search_results.articles) == 0:
            yield RunResponse(
                event=RunEvent.workflow_completed,
                content=f"Sorry, could not find any articles on the topic: {topic}",
            )
            return

        yield from self.write_blog_post(topic, search_results)

    def get_cached_blog_post(self, topic: str) -> Optional[str]:
        logger.info("Checking if cached blog post exists")
        return self.session_state.get("blog_posts", {}).get(topic)

    def add_blog_post_to_cache(self, topic: str, blog_post: Optional[str]):
        logger.info(f"Saving blog post for topic: {topic}")
        self.session_state.setdefault("blog_posts", {})
        self.session_state["blog_posts"][topic] = blog_post

    def get_search_results(self, topic: str) -> Optional[SearchResults]:
        MAX_ATTEMPTS = 3
        for attempt in range(MAX_ATTEMPTS):
            try:
                searcher_response: RunResponse = self.searcher.run(topic)
                if not searcher_response or not searcher_response.content:
                    logger.warning(f"Attempt {attempt + 1}/{MAX_ATTEMPTS}: Empty searcher response")
                    continue
                if not isinstance(searcher_response.content, SearchResults):
                    logger.warning(f"Attempt {attempt + 1}/{MAX_ATTEMPTS}: Invalid response type")
                    continue
                logger.info(f"Found {len(searcher_response.content.articles)} articles on attempt {attempt + 1}")
                return searcher_response.content
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{MAX_ATTEMPTS} failed: {str(e)}")
        logger.error(f"Failed to get search results after {MAX_ATTEMPTS} attempts")
        return None

    def write_blog_post(self, topic: str, search_results: SearchResults) -> Iterator[RunResponse]:
        logger.info("Writing blog post")
        writer_input = {"topic": topic, "articles": [article.model_dump() for article in search_results.articles]}
        yield from self.writer.run(json.dumps(writer_input, indent=4), stream=True)
        self.add_blog_post_to_cache(topic, self.writer.run_response.content)

DB_PATH = "user_data.db"

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS blog_posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            topic TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            query TEXT,
            results TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    return conn

def signup_user(username: str, password: str) -> bool:
    conn = init_db()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode(), hashed_password.encode())

def check_user(username: str, password: str) -> bool:
    conn = init_db()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (username))
    user = c.fetchone()
    conn.close()
    if user and verify_password(password, user['password_hash']):
        return user is not None

def get_cached_blog_post_db(username: str, topic: str) -> Optional[str]:
    conn = init_db()
    c = conn.cursor()
    c.execute(
        "SELECT content FROM blog_posts WHERE username=? AND topic=? ORDER BY timestamp DESC LIMIT 1",
        (username, topic),
    )
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

def add_blog_post_db(username: str, topic: str, content: str):
    conn = init_db()
    c = conn.cursor()
    c.execute(
        "INSERT INTO blog_posts (username, topic, content) VALUES (?, ?, ?)",
        (username, topic, content),
    )
    conn.commit()
    conn.close()

def get_blog_posts_db(username: str):
    conn = init_db()
    c = conn.cursor()
    c.execute(
        "SELECT topic, content, timestamp FROM blog_posts WHERE username=? ORDER BY timestamp DESC",
        (username,),
    )
    rows = c.fetchall()
    conn.close()
    return rows

def add_search_history_db(username: str, query: str, to_store: str):
    conn = init_db()
    c = conn.cursor()
    c.execute(
        "INSERT INTO search_history (username, query, results) VALUES (?, ?, ?)",
        (username, query, to_store),
    )
    conn.commit()
    conn.close()

def get_search_history_db(username: str):
    conn = init_db()
    c = conn.cursor()
    c.execute(
        "SELECT query, results, timestamp FROM search_history WHERE username=? ORDER BY timestamp DESC",
        (username,),
    )
    rows = c.fetchall()
    conn.close()
    return rows

def generate_blog_post(username: str, topic: str) -> str:
    cached = get_cached_blog_post_db(username, topic)
    if cached:
        st.info("Using cached blog post.")
        return cached

    workflow_session_id = f"generate-blog-post-{username}-{topic.replace(' ', '-')}"
    blog_generator = BlogPostGenerator(
        session_id=workflow_session_id,
        storage=SqlWorkflowStorage(
            table_name="generate_blog_post_workflows",
            db_file="workflows.db",
        ),
    )

    blog_content = ""
    with st.spinner("Generating blog post..."):
        responses = blog_generator.run(topic=topic, use_cache=True)
        for response in responses:
            blog_content += response.content + "\n"
    add_blog_post_db(username, topic, blog_content)
    return blog_content

def main():
    st.title("Blog Post Generator & News Search")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        st.sidebar.header("Login / Sign Up")
        auth_mode = st.sidebar.radio("Select Mode", options=["Login", "Sign Up"])
        login_username = st.sidebar.text_input("Username", key="login_username")
        login_password = st.sidebar.text_input("Password", key="login_password", type="password")

        if auth_mode == "Login":
            if st.sidebar.button("Login"):
                if check_user(login_username, login_password):
                    st.session_state.logged_in = True
                    st.session_state.username = login_username
                    st.success("Logged in!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        else:  
            if st.sidebar.button("Sign Up"):
                if signup_user(login_username, login_password):
                    st.session_state.logged_in = True
                    st.session_state.username = login_username
                    st.success("Signed up and logged in!")
                else:
                    st.error("Username already exists, try a different one.")

    if not st.session_state.logged_in:
        st.warning("Please log in or sign up via the sidebar to access the features.")
        return

    st.sidebar.title("Blog Generator and News Reader")

    with st.container():
        st.sidebar.subheader(f"Welcome Back , {st.session_state.username} !!")

    st.sidebar.subheader("View History")
    history_type = st.sidebar.radio("Select History to view", options=["Blog Generation", "Web Search"])
    if history_type == "Blog Generation":
        with st.sidebar:
            st.write("Blog Generation History")
            posts = get_blog_posts_db(st.session_state.username)
            if posts:
                for post_topic, content, timestamp in posts:
                    with st.expander(f"Topic: {post_topic} (Created on {timestamp})"):
                        st.markdown(content)
            else:
                st.write("No blog posts found.")

    else:
        with st.sidebar:
            st.write("Web Search History")
            history = get_search_history_db(st.session_state.username)
            if history:
                for past_query, results, timestamp in history:
                    with st.expander(f"Query: {past_query} (Searched on {timestamp})"):
                        try:
                            results_dict = json.loads(results)
                            titles = results_dict["Title"]
                            descriptions = results_dict["Description"]
                            urls = results_dict["URL"]

                            for i in range(len(titles)):
                                st.write(f"- {titles[i]}")
                                st.write(f"- {descriptions[i]}")
                                st.write(f"- {urls[i]}")
                                st.markdown('---')
                        except Exception as e:
                            st.write(f"Error parsing results. {str(e)}")
            else:
                st.write("No search history found.")

    if st.session_state.logged_in:
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()

    tab_blog, tab_search = st.tabs(["Blog Creation", "News Search"])

    with tab_blog:
        st.header("Generate a Blog Post")
        topic = st.text_input("Enter a blog post topic", key="blog_topic")
        if st.button("Generate Blog Post"):
            if not topic:
                st.error("Please enter a topic for the blog post.")
            else:
                blog_post = generate_blog_post(st.session_state.username, topic)
                st.markdown(blog_post)

    with tab_search:
        st.header("Search for News Articles")
        query = st.text_input("Enter your search query", key="search_query")
        num = st.slider(label="Enter the number of searches", min_value=1,max_value=20, value=5, key="search_num")

        if st.button("Search News"):
            if not query:
                st.error("Please enter a search query.")
            else:
                try:
                    url=f'https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={n_api}&language=en'
                    r=requests.get(url)
                    r=r.json()

                    to_store = {
                        "Title" : [],
                        "Description" : [],
                        "URL" : []
                    }

                    articles=r['articles']
                    for article in articles[:num]:
                        st.header(article['title'])
                        st.write(f"<h5 style=''> Published at : {article['publishedAt']}</h5>",unsafe_allow_html=True)
                        to_store["Title"].append(article['title'])
                        
                        if article['author']:
                            st.write(article['author'])

                        if article['description']== None:
                            st.write("Refer The Link")
                            st.write(f"{article['url']}")
                            to_store["URL"].append(article["url"])

                        else:
                            st.write(article['source']['name'])
                            st.write(article['description'])
                            st.write(f"See More :  {article['url']}")

                            to_store["Description"].append(article['description'])
                            to_store["URL"].append(article['url'])

                            try:
                                st.image(article['urlToImage'])
                            except AttributeError:
                                st.write("IMAGE IS NOT AVAILABLE")
                            else:
                                pass

                    add_search_history_db(st.session_state.username, query, json.dumps(to_store))
                
                except Exception as e:
                    st.error("No results found. Please try again.")

if __name__ == "__main__":
    main()
