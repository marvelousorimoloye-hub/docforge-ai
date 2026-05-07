#tests\test_fire.py
import os
from firecrawl import FirecrawlApp
from core.config import FIRECRAWL_API_KEY
# Manually paste your key here for a one-time test
app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

try:
    print("🚀 Testing Firecrawl...")
    # Scrape a simple, stable site
    result = app.scrape("https://agfundernews.com/a-brussels-moat-can-european-regulation-work-for-founders",only_main_content=True, formats=['markdown'], include_tags=['main', 'article', '.ecl-container'] )
    
    if result.markdown:
        print(f"✅ SUCCESS! Scraped {len(result.markdown)} characters.")
        print(f"Snippet: {result.markdown[:1500]}...")
    else:
        print("❌ FAILED: No content returned.")
        print(f"Full Response: {result}")
except Exception as e:
    print(f"❌ API ERROR: {str(e)}")
