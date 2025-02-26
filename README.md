# Blog Post Generator & News Search

This is a Streamlit-based web application that generates blog posts using Google's Gemini API and fetches related news articles via DuckDuckGo search. The application also supports user authentication and maintains a database of generated blog posts and search history.

## Features

- **User Authentication:** Sign up and log in using a secure password hashing mechanism.
- **Blog Post Generation:** Uses Gemini API to generate high-quality blog posts based on news articles.
- **News Search:** Retrieves relevant news articles using DuckDuckGo search.
- **Database Storage:** Stores user data, blog posts, and search history in an SQLite database.
- **Session Management:** Caches generated blog posts for faster retrieval.

## Installation

### Prerequisites

- Python 3.8+
- `pip` (Python package manager)

### Setup

1. Clone this repository:
   ```sh
   git clone https://github.com/your-repo/blog-post-generator.git
   cd blog-post-generator
   ```
2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   - Create a `.env` file and add:
     ```
     GOOGLE_API_KEY=your_gemini_api_key
     NEWS_API=your_news_api_key
     ```

## Usage

1. Run the application:
   ```sh
   streamlit run app.py
   ```
2. Open the application in your browser and log in or sign up.
3. Enter a topic to generate a blog post and view related news articles.

## Database Structure

The application uses SQLite with the following tables:

- `users`: Stores user credentials.
- `blog_posts`: Stores generated blog posts.
- `search_history`: Stores previous news search results.

## Technologies Used

- **Python**: Backend logic
- **Streamlit**: Web interface
- **SQLite**: Database storage
- **Google Gemini API**: AI-generated blog posts
- **DuckDuckGo**: News search
