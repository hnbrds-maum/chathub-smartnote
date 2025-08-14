# html_fetcher.py
from typing import Optional
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright

class HTTPStatusError(Exception):
    def __init__(self, url: str, status: int):
        super().__init__(f"{status} @ {url}")
        self.url = url
        self.status = status        


def fetch_rendered_html(url: str,
                        wait_selector: Optional[str] = None,
                        timeout: int = 10000) -> str:
    """전체 JS 렌더링 결과 HTML 반환"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        response = page.goto(url, wait_until="networkidle", timeout=timeout)
        status = response.status if response else 0
        if status != 200:
            browser.close()
            raise HTTPStatusError(url, status)
        if wait_selector:
            page.wait_for_selector(wait_selector, timeout=timeout)
        html = page.content()        # 최종 DOM 스냅샷
        browser.close()
        return html

async def afetch_rendered_html(url: str,
                              wait_selector: Optional[str] = None,
                              timeout: int = 10_000) -> str:
    """전체 JS 렌더링 결과 HTML 반환 (비동기 버전)"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        response = await page.goto(url, wait_until="networkidle", timeout=timeout)
        status = response.status if response else 0
        if status != 200:
            await browser.close()
            raise HTTPStatusError(url, status)
        if wait_selector:
            await page.wait_for_selector(wait_selector, timeout=timeout)
        html = await page.content()
        await browser.close()
        return html