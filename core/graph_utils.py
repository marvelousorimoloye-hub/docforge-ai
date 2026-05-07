#graph_utils.py
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse
from sentence_transformers import CrossEncoder
from firecrawl import FirecrawlApp
import time
from core.config import FIRECRAWL_API_KEY
from fpdf import FPDF
from datetime import datetime
import os
from typing import List
import voyageai

# --- ETHICAL SCRAPING UTILS ---
def is_allowed_by_robots(url: str, user_agent="DocForgeBot/1.0"):
    """Politely checks robots.txt before scraping."""
    try:
        parsed_url = urlparse(url)
        robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except:
        return True # Default to True if robots.txt is missing

def ethical_scrape(url: str):
    """Executes an ethical scrape with rate limiting."""
    if not is_allowed_by_robots(url):
        return f"[SKIPPED] robots.txt disallows scraping for: {url}"
    
    # Ethical delay to avoid server strain
    time.sleep(2)
    
    try:
        app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
        # Pass arguments directly to the scrape method
        result = app.scrape(
            url, 
            formats=['markdown'], 
            only_main_content=True
        )
    
        # result.markdown is a string, so we just check if it exists
        if result.markdown:
            print(f"✅ SUCCESS! Scraped {len(result.markdown)} characters.")
            print(f"Snippet: {result.markdown[:1500]}...")
            return result.markdown
        return "No content extracted."

    except Exception as e:
        return f"[SCRAPE ERROR] {url}: {str(e)}"


# --- GRAPH NODES ---



# Set timeout for the fallback model download
os.environ["HTTP_TIMEOUT"] = "60"

# Initialize Voyage AI client 
vo = voyageai.Client()

# Initialize Fallback Reranker
fallback_reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_content(query: str, text_chunks: List[str], top_k: int = 5):
    """Uses Voyage AI to rerank chunks, falls back to Cross-Encoder if API fails."""
    if not text_chunks:
        return ""

    try:
        # 1. Attempt Voyage AI Reranking (Primary)
        result = vo.rerank(query, text_chunks, model="rerank-2", top_k=top_k)
        
        # Voyage returns objects with index and relevance_score
        relevant_chunks = [r.document for r in result.results if r.relevance_score > 0.1]
        print("Reranking successful using Voyage AI.")

    except Exception as e:
        # 2. Fallback to Local Cross-Encoder
        print(f"Voyage AI failed or Key missing: {e}. Falling back to Cross-Encoder...")
        
        pairs = [[query, chunk] for chunk in text_chunks]
        scores = fallback_reranker.predict(pairs)
        
        # Sort by score
        scored_chunks = sorted(zip(scores, text_chunks), reverse=True, key=lambda x: x[0])
        relevant_chunks = [chunk for score, chunk in scored_chunks[:top_k] if score > 0]

    return "\n\n".join(relevant_chunks)


class DocForgeReportGenerator:
    """Handles beautiful PDF report generation"""

    @staticmethod
    def generate_pdf(report_text: str, query: str, analysis: str = "") -> bytes:
        class PDF(FPDF):
            def header(self):
                self.set_font("Arial", "B", 15)
                self.cell(0, 10, "DocForge AI - Compliance & Risk Report", ln=True, align="C")
                self.ln(5)
                self.set_font("Arial", "", 10)
                self.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
                self.ln(8)

            def footer(self):
                self.set_y(-15)
                self.set_font("Arial", "I", 8)
                self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

        pdf = PDF()
        pdf.alias_nb_pages()
        pdf.add_page()

        # Title & Query
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 12, "Analysis Query", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 8, query)
        pdf.ln(12)

        # Table of Contents
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Table of Contents", ln=True)
        pdf.set_font("Arial", "", 12)
        for item in ["1. Executive Summary", "2. Key Findings", "3. Risk Assessment", 
                     "4. Detailed Analysis", "5. Recommendations"]:
            pdf.cell(0, 8, item, ln=True)
        pdf.ln(15)

        # Main Content
        pdf.set_font("Arial", "", 11)
        sections = report_text.split("\n\n")
        
        for section in sections:
            if section.strip():
                # Handle bold-like sections
                if any(x in section.lower() for x in ["summary", "findings", "risk", "recommend"]):
                    pdf.set_font("Arial", "B", 12)
                    clean = section.replace("**", "").strip()
                    pdf.multi_cell(0, 8, clean)
                    pdf.ln(6)
                    pdf.set_font("Arial", "", 11)
                else:
                    pdf.multi_cell(0, 8, section)
                    pdf.ln(4)

        # Risk Heatmap Page
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Risk Heatmap & Summary", ln=True)
        pdf.ln(15)

        pdf.set_font("Arial", "", 11)
        pdf.cell(0, 8, "• High Risk Items   : [          ]", ln=True)
        pdf.cell(0, 8, "• Medium Risk Items : [          ]", ln=True)
        pdf.cell(0, 8, "• Low Risk Items    : [          ]", ln=True)
        pdf.ln(10)
        
        pdf.multi_cell(0, 8, 
            "This section can later include actual risk scores, color-coded tables, "
            "or compliance checklists.")

        return pdf.output(dest="S").encode("latin-1")
    
    # core/graph_utils.py
# ... (your existing code: ethical_scrape, rerank_content, DocForgeReportGenerator, etc.)

from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log, retry_if_exception_type
import logging

# Setup logger for retry messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def with_retry(max_retries: int = 3, min_delay: int = 2):
    """
    Production-ready retry decorator using Tenacity.
    - Exponential backoff
    - Clean logging
    - Easy to customize per node
    """
    return retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=min_delay, min=min_delay, max=30),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,           # Re-raise the original exception after retries fail
    )

# core/graph_utils.py
# ... (existing code: DocForgeReportGenerator, with_retry, etc.)

import time
import logging
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """
    Circuit Breaker Pattern for AI Agent resilience.
    Prevents cascading failures when external services are down.
    """
    
    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def __call__(self, func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    logger.info(f"CircuitBreaker '{self.name}' → Half-Open (testing)")
                else:
                    logger.warning(f"CircuitBreaker '{self.name}' is OPEN. Skipping call.")
                    raise Exception(f"Circuit breaker '{self.name}' is OPEN")

            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise

        return wrapper

    def _on_success(self):
        self.failure_count = 0
        if self.state != "CLOSED":
            logger.info(f"CircuitBreaker '{self.name}' → CLOSED (recovered)")
            self.state = "CLOSED"

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(f"CircuitBreaker '{self.name}' → OPEN after {self.failure_count} failures")


# Global Circuit Breakers
pinecone_circuit = CircuitBreaker("Pinecone", failure_threshold=4, recovery_timeout=45)
groq_circuit = CircuitBreaker("Groq", failure_threshold=5, recovery_timeout=30)
tavily_circuit = CircuitBreaker("Tavily", failure_threshold=3, recovery_timeout=20)
scrape_circuit = CircuitBreaker("WebScrape", failure_threshold=6, recovery_timeout=60)