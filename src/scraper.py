"""Amazon product review scraper for deceptive review detection."""

import logging
import re
import time
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

LOGGER = logging.getLogger(__name__)

# Realistic browser headers to reduce bot-detection rejections
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0",
    "DNT": "1",
}

_TIMEOUT = 15  # seconds


def extract_asin(url: str) -> Optional[str]:
    """Return the 10-char ASIN from any Amazon product URL, or None."""
    for pattern in (
        r"/dp/([A-Z0-9]{10})",
        r"/product/([A-Z0-9]{10})",
        r"[?&]asin=([A-Z0-9]{10})",
        r"/gp/product/([A-Z0-9]{10})",
    ):
        m = re.search(pattern, url, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def _get_domain(url: str) -> str:
    """Extract the amazon.* domain from a URL (defaults to amazon.com)."""
    m = re.search(r"(amazon\.[a-z.]{2,6})", url, re.IGNORECASE)
    return m.group(1).lower() if m else "amazon.com"


def _fetch_html(session: requests.Session, url: str) -> BeautifulSoup:
    """Fetch a URL and return a BeautifulSoup object; raise on failure."""
    try:
        resp = session.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    except requests.RequestException as exc:
        raise ConnectionError(f"Network error while contacting Amazon: {exc}") from exc

    if resp.status_code == 503:
        raise ConnectionError("Amazon is rate-limiting requests (503). Wait a moment and try again.")
    if resp.status_code != 200:
        raise ConnectionError(f"Amazon returned HTTP {resp.status_code}.")

    soup = BeautifulSoup(resp.content, "lxml")
    page_text = soup.get_text(" ", strip=True).lower()
    if "enter the characters you see below" in page_text or "captcha" in page_text:
        raise PermissionError(
            "Amazon returned a CAPTCHA page. "
            "Try opening the product page in your browser first, then retry."
        )
    return soup


def _parse_reviews(soup: BeautifulSoup, asin: str) -> List[Dict]:
    """Extract review records from a parsed reviews page."""
    review_divs = soup.find_all("div", {"data-hook": "review"})
    records = []
    for div in review_divs:
        # --- review text ---
        body = div.find("span", {"data-hook": "review-body"})
        text = body.get_text(" ", strip=True) if body else ""
        if not text:
            continue

        # --- star rating ---
        star_el = div.find("i", {"data-hook": "review-star-rating"}) or \
                  div.find("i", {"data-hook": "cmps-review-star-rating"})
        rating = 5
        if star_el:
            alt = star_el.find("span", class_="a-icon-alt")
            if alt:
                try:
                    rating = int(float(alt.get_text().split()[0]))
                except (ValueError, IndexError):
                    pass

        # --- verified purchase ---
        verified = div.find("span", {"data-hook": "avp-badge"}) is not None

        # --- reviewer name ---
        name_el = div.find("span", class_="a-profile-name")
        reviewer_name = name_el.get_text(strip=True) if name_el else "Anonymous"

        # --- review title ---
        title_el = div.find("a", {"data-hook": "review-title"}) or \
                   div.find("span", {"data-hook": "review-title"})
        title = title_el.get_text(" ", strip=True) if title_el else ""

        records.append({
            "reviewer_name": reviewer_name,
            "title": title,
            "review_text": text,
            "rating": rating,
            "verified_purchase": int(verified),
            "reviewer_review_count": 1,
            "product_id": asin,
        })
    return records


def scrape_amazon_reviews(url: str, max_reviews: int = 15) -> List[Dict]:
    """
    Scrape up to *max_reviews* reviews from an Amazon product URL.

    Returns a list of dicts with keys:
        reviewer_name, title, review_text, rating,
        verified_purchase, reviewer_review_count, product_id

    Raises:
        ValueError        – URL contains no recognisable ASIN
        ConnectionError   – network failure or HTTP error
        PermissionError   – Amazon CAPTCHA / bot-check page
        RuntimeError      – page loaded but no reviews were found
    """
    asin = extract_asin(url)
    if not asin:
        raise ValueError(
            "Could not find an ASIN in the URL. "
            "Make sure the link is a valid Amazon product page "
            "(e.g. https://www.amazon.com/dp/B08N5WRWNW)."
        )

    domain = _get_domain(url)
    session = requests.Session()

    reviews: List[Dict] = []
    pages_needed = -(-max_reviews // 10)  # ceiling division, 10 reviews per page

    for page_num in range(1, pages_needed + 1):
        reviews_url = (
            f"https://www.{domain}/product-reviews/{asin}/"
            f"?pageNumber={page_num}&sortBy=recent&reviewerType=all_reviews"
        )
        LOGGER.info("Fetching reviews page %d: %s", page_num, reviews_url)

        # Warm up cookies on the first page only
        if page_num == 1:
            try:
                session.get(
                    f"https://www.{domain}/dp/{asin}",
                    headers=_HEADERS,
                    timeout=_TIMEOUT,
                )
                time.sleep(1.2)
            except requests.RequestException:
                pass  # best-effort cookie warm-up

        soup = _fetch_html(session, reviews_url)
        batch = _parse_reviews(soup, asin)

        if not batch:
            if page_num == 1:
                raise RuntimeError(
                    "No reviews were found on the page. "
                    "Amazon may have blocked the request, or this product has no reviews yet."
                )
            break  # earlier pages had data; we've exhausted them

        reviews.extend(batch)
        if len(reviews) >= max_reviews:
            break

        time.sleep(1.0)  # polite delay between page requests

    return reviews[:max_reviews]
