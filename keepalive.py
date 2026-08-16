from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("https://meanreversiontool.streamlit.app/", wait_until="domcontentloaded", timeout=120000)
    page.wait_for_timeout(5000)
    btn = page.get_by_role("button", name="Yes, get this app back up!")
    if btn.count() > 0:
        btn.click()
        page.wait_for_timeout(60000)
    browser.close()
