"""Headless browser QA for the actual locally forwarded dashboard."""
from pathlib import Path
from playwright.sync_api import sync_playwright

root=Path(__file__).resolve().parents[1]
errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto('http://127.0.0.1:55318/',wait_until='networkidle')
    page.wait_for_function("document.getElementById('connection').textContent.includes('实时连接')")
    print('title',page.title(),'cards',page.locator('.run').count(),'browser_errors',errors)
    page.screenshot(path=str(root/'Model/mae_dashboard_desktop.png'),full_page=True)
    if page.locator('option[value="M1_LoRA_r8_b16_s17"]').count():
        page.select_option('#runSelect','M1_LoRA_r8_b16_s17')
        page.wait_for_function("document.getElementById('detail').textContent.includes('有效batch 16')")
        page.screenshot(path=str(root/'Model/mae_dashboard_hybrid.png'),full_page=True)
        print('hybrid_batch16_label_verified')
    if page.locator('.run').count():
        page.select_option('#metric','2');page.locator('.run').last.click()
        page.set_viewport_size({'width':390,'height':844})
        page.screenshot(path=str(root/'Model/mae_dashboard_mobile.png'),full_page=True)
    browser.close()
if errors:raise RuntimeError(errors)
