import openai
import re
import requests
from bs4 import BeautifulSoup
from config import embed_model, index

course_catalog_urls = [
    "https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/advanced-intelligent-manufacturing-ms/#programrequirementstext",
    "https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/data-analytics-engineering-ms/#programrequirementstext",
    "https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/data-analytics-engineering-online-ms/#programrequirementstext",
    "https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/human-factors-mshf/#programrequirementstext",
    "https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/robotics-ms/#programrequirementstext",
    "https://catalog.northeastern.edu/graduate/engineering/electrical-computer/semiconductor-engineering-ms/#programrequirementstext",
    "https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/industrial-engineering-msie/#programrequirementstext",
    "https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/engineering-management-msem/#programrequirementstext",
    "https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/energy-systems-msenes/#programrequirementstext",
    "https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/energy-systems-msenes-academic-link-program/#programrequirementstext",
    "https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/mechanical-engineering-concentration-general-msme/#programrequirementstext",
    "https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/mechanical-engineering-concentration-mechanics-design-msme/#programrequirementstext",
    "https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/mechanical-engineering-concentration-material-science-msme/#programrequirementstext",
    "https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/mechanical-engineering-concentration-mechatronics-msme/#programrequirementstext",
    "https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/mechanical-engineering-concentration-thermofluids-msme/#programrequirementstext",
    "https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/operations-research-msor/#programrequirementstext"
]

# In-memory session memory (stores question-answer pairs)
session_memory = []

def estimate_max_tokens(query, context, base_tokens=150, max_limit=500):
    context_text = "\n".join(context)
    extra_tokens = len(context_text) // 4
    return min(base_tokens + extra_tokens, max_limit)

# ---------------- URL Helpers ----------------

def extract_urls(text):
    return re.findall(r'https?://\S+', text)

def verify_urls(url_list):
    valid_urls, invalid_urls = [], []
    for url in url_list:
        try:
            r = requests.head(url, allow_redirects=True, timeout=5)
            if r.status_code == 200:
                valid_urls.append(url)
            else:
                invalid_urls.append((url, r.status_code))
        except Exception as e:
            invalid_urls.append((url, str(e)))
    return valid_urls, invalid_urls

def verify_urls_in_text(text):
    urls = extract_urls(text)
    if not urls:
        return text
    valid_urls, invalid_urls = verify_urls(urls)
    for url, status in invalid_urls:
        text = text.replace(url, f"{url} (invalid: {status})")
    return text

# ---------------- Course Query Handling ----------------

def is_course_related_query(query):
    """Check if the query is related to courses or programs."""
    course_keywords = [
        'course', 'program', 'curriculum', 'degree', 'major', 'minor',
        'concentration', 'requirement', 'credit', 'class', 'prerequisite',
        'elective', 'semester', 'ms', 'msie', 'msem', 'msenes', 'msme', 'msor'
    ]
    
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()
    
    # Check if any course keyword is in the query
    return any(keyword in query_lower for keyword in course_keywords)

def course_catalog_agent(query):
    """Agent that selects the appropriate course catalog URL based on the query."""
    prompt = f"""
    Based on the following user query about Northeastern University courses or programs, select the MOST RELEVANT URL from the list:
    
    User Query: "{query}"
    
    Available catalog URLs:
    1. Advanced Intelligent Manufacturing MS (AI Manufacturing): https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/advanced-intelligent-manufacturing-ms/#programrequirementstext
    2. Data Analytics Engineering MS (DAE): https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/data-analytics-engineering-ms/#programrequirementstext
    3. Data Analytics Engineering Online MS: https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/data-analytics-engineering-online-ms/#programrequirementstext
    4. Human Factors MSHF: https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/human-factors-mshf/#programrequirementstext
    5. Robotics MS: https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/robotics-ms/#programrequirementstext
    6. Semiconductor Engineering MS: https://catalog.northeastern.edu/graduate/engineering/electrical-computer/semiconductor-engineering-ms/#programrequirementstext
    7. Industrial Engineering MSIE: https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/industrial-engineering-msie/#programrequirementstext
    8. Engineering Management MSEM: https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/engineering-management-msem/#programrequirementstext
    9. Energy Systems MSENES: https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/energy-systems-msenes/#programrequirementstext
    10. Energy Systems MSENES Academic Link: https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/energy-systems-msenes-academic-link-program/#programrequirementstext
    11. Mechanical Engineering (General) MSME: https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/mechanical-engineering-concentration-general-msme/#programrequirementstext
    12. Mechanical Engineering (Mechanics Design) MSME: https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/mechanical-engineering-concentration-mechanics-design-msme/#programrequirementstext
    13. Mechanical Engineering (Material Science) MSME: https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/mechanical-engineering-concentration-material-science-msme/#programrequirementstext
    14. Mechanical Engineering (Mechatronics) MSME: https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/mechanical-engineering-concentration-mechatronics-msme/#programrequirementstext
    15. Mechanical Engineering (Thermofluids) MSME: https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/mechanical-engineering-concentration-thermofluids-msme/#programrequirementstext
    16. Operations Research MSOR: https://catalog.northeastern.edu/graduate/engineering/mechanical-industrial/operations-research-msor/#programrequirementstext
    
    Return only the URL that's most relevant to the query, no other text.
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Course catalog agent error: {e}")
        # Return a default URL if there's an error
        return course_catalog_urls[0]
    

def extract_rich_text(soup):
    content = []

    # Track latest heading to attach context (like "Required Courses")
    last_heading = ""

    # Extract heading blocks and paragraphs
    for tag in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        text = tag.get_text(strip=True)
        if not text:
            continue

        if tag.name in ["h1", "h2", "h3"]:
            last_heading = text
            content.append(f"### {text}")
        elif tag.name == "li":
            content.append(f"- {text}")
        else:
            content.append(text)

    # Extract course-related tables
    for table in soup.find_all("table"):
        table_rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True).replace("\xa0", " ") for td in tr.find_all("td")]
            if cells:
                table_rows.append(cells)

        # Decide structure based on number of cells
        if table_rows:
            # Add heading if previous text was relevant
            if last_heading and any(kw in last_heading.lower() for kw in ["course", "requirement", "curriculum", "core", "elective"]):
                content.append(f"#### {last_heading} Table")

            # Markdown table for 2 or 3 columns
            if all(len(row) == 3 for row in table_rows):
                content.append("Course Code | Course Title | Credits")
                content.append("--- | --- | ---")
                for row in table_rows:
                    content.append(" | ".join(row))
            elif all(len(row) == 2 for row in table_rows):
                content.append("Course Code | Course Title")
                content.append("--- | ---")
                for row in table_rows:
                    content.append(" | ".join(row))
            else:
                # Fallback: plain bullets
                for row in table_rows:
                    content.append(f"- {' – '.join(row)}")

    # Debug preview
    final_text = "\n".join(content)
    return final_text

def scrape_course_catalog(url):
    """Scrape content from the course catalog URL using rich text extraction"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get program title
        title_elem = soup.find('h1')
        program_title = title_elem.get_text(strip=True) if title_elem else "Program Requirements"
        
        # Find the main content section - try multiple possible container IDs
        content_section = None
        for container_id in ['programrequirementstextcontainer', 'programrequirementstext']:
            content_section = soup.find('div', {'id': container_id})
            if content_section:
                break
        
        # Fallback to other common containers if specific ones not found
        if not content_section:
            for container_class in ['page_content', 'main-content', 'content-wrapper']:
                content_section = soup.find('div', {'class': container_class})
                if content_section:
                    break
        
        if not content_section:
            return {
                "title": program_title,
                "content": f"Program: {program_title}\n\nCould not find program requirements section. Please check the URL directly.",
                "url": url
            }
        
        # Extract rich text content
        extracted_content = extract_rich_text(content_section)
        
        # Format the final content
        formatted_content = f"Program: {program_title}\n\n{extracted_content}"
        
        return {
            "title": program_title,
            "content": formatted_content,
            "url": url
        }
    except Exception as e:
        print(f"Scraping error for {url}: {e}")
        return {
            "title": "Error",
            "content": f"Failed to scrape content from {url}. Error: {str(e)}",
            "url": url
        }

# ---------------- GPT Agents ----------------

def fallback_scraper_agent(query):
    prompt = f"""
Search the web for detailed information about: '{query}' in the context of Northeastern University. Provide a concise summary.
Also provide a helpful link.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        raw = response["choices"][0]["message"]["content"].strip()
        return verify_urls_in_text(raw)
    except Exception as e:
        print("Fallback scraper error:", e)
        return ""

def query_optimizer_agent(query, chat_history=None):
    history_block = "\n".join([f"Previous Q: {turn['question']}" for turn in chat_history[-5:]]) if chat_history else ""
    prompt = f"""
You are an intelligent assistant specializing in queries related to Northeastern University.
{history_block}

Now improve the following query for clarity and relevance. Keep in context the history block while optimising the query
in case the query is a follow-up question of the previous query. If the question/query is of a different program, do not use
previous context. If it's related to a different topic, do not use previous context.:
"{query}"
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=60
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("Query optimizer error:", e)
        return query

def retrieve_context(query, top_k=3, threshold=0.7):
    query_embedding = embed_model.encode(query).tolist()
    result = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    context = []
    if result and "matches" in result:
        for match in result["matches"]:
            if match.get("score", 0) >= threshold:
                context.append(match["metadata"].get("combined_text", ""))
    return context

def rag_agent(query, context, chat_history=""):
    if not context or all(not c.strip() for c in context):
        return ("I'm sorry, I don't have sufficient information about this topic. "
                "Please visit [FAQs](https://northeastern.edu/faqs) or contact [support@northeastern.edu](mailto:support@northeastern.edu).")

    context_text = "\n\n".join(context)
    prompt = (
        f"Chat History:\n{chat_history}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        "Please provide a clear, concise and helpful answer about Northeastern University. "
        "Keep in context the chat history as well when answering questions. "
        "Format your answer in Markdown with headings and bullet points when needed. "
        "Include valid links when applicable. If no valid context, say so. "
        "If the answer includes links, add: "
        "'If the above link doesn't work or you need updated info, visit the official [Northeastern program page](https://graduate.northeastern.edu/programs/) or use the [search function](https://www.northeastern.edu/search/)'."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for Northeastern University."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=estimate_max_tokens(query, context)
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("RAG error:", e)
        return "I'm sorry, I couldn't generate a response. Please contact support."

# ---------------- Memory Retrieval ----------------

def get_question_by_index(ordinal: str) -> str:
    ordinal_map = {
        "first": 1, "1st": 1,
        "second": 2, "2nd": 2,
        "third": 3, "3rd": 3,
        "fourth": 4, "4th": 4,
        "fifth": 5, "5th": 5,
    }
    try:
        index_num = int(ordinal) if ordinal.isdigit() else ordinal_map.get(ordinal.lower(), 1)
    except Exception:
        index_num = 1
    if 1 <= index_num <= len(session_memory):
        return session_memory[index_num - 1]["question"]
    else:
        return "No such question found in this session."

# ---------------- Main Chat Function ----------------

def process_chat(user_query: str, chat_history: list[str] = []) -> str:
    print(f"\n[PROCESS_CHAT] 🔹 Received user query: {user_query}")

    # Check for request for a previous question
    match = re.search(r"what was my (\w+)[\s-]*question", user_query.lower())
    if match:
        return f"Your requested question: {get_question_by_index(match.group(1))}"

    # Step 1: Use optimizer with session memory context
    optimized_query = query_optimizer_agent(user_query, chat_history=session_memory)
    print(f"[OPTIMIZER] ✨ Optimized query: {optimized_query}")
    
    # Step 2: Check if this is a course-related query
    if is_course_related_query(optimized_query):
        print("[COURSE] 📚 Detected course-related query")
        
        # Get relevant course catalog URL
        catalog_url = course_catalog_agent(optimized_query)
        print(f"[COURSE] 🔗 Selected catalog URL: {catalog_url}")
        
        # Scrape the content from the URL
        scraped_data = scrape_course_catalog(catalog_url)
        print(f"[SCRAPER] 🌐 Scraped content from: {scraped_data['title']}")
        
        # Format the scraped content for the RAG agent
        context_docs = [
            f"Title: {scraped_data['title']}\n\n"
            f"Content: {scraped_data['content']}\n\n"
            f"Source: {scraped_data['url']}"
        ]
    else:
        # Regular flow for non-course queries
        # Step 3: Try Pinecone retrieval
        context_docs = retrieve_context(optimized_query)
        print(f"[PINECONE] 📚 Retrieved {len(context_docs)} documents")

        # Step 4: Fallback if nothing found
        if not context_docs or all(not c.strip() for c in context_docs):
            print("[FALLBACK] 🪄 Using GPT fallback")
            context_docs = [fallback_scraper_agent(optimized_query)]

    # Step 5: Build chat history (last 5 rounds)
    formatted_chat_history = "\n".join([
        f"User: {msg['question']}\nAssistant: {msg['answer']}"
        for msg in session_memory[-5:]
    ])

    # Step 6: Generate answer using RAG
    final_response = rag_agent(optimized_query, context_docs, formatted_chat_history)

    # Step 7: Store interaction
    session_memory.append({
        "question": user_query,
        "answer": final_response
    })

    print(f"[RAG] ✅ Final response length: {len(final_response)}")
    return final_response