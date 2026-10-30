"""Browser QA for the actual sweep monitoring page."""
from pathlib import Path
from playwright.sync_api import sync_playwright

errors=[];root=Path(__file__).resolve().parents[1]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1100})
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto('http://127.0.0.1:55318/',wait_until='networkidle')
    page.wait_for_function("document.querySelectorAll('.run').length===17")
    for name,batch,stage in [('M1_AllLoRA_b16_s17',16,'all_lora'),('M1_Full_b16_s29',16,'full'),('M1_AllLoRA_b8_s29',8,'all_lora'),('M1_HybridLoRA_b8_s17',8,'hybrid')]:
        page.select_option('#runSelect',name)
        page.wait_for_function("([name,batch])=>document.querySelector('#selectedTitle').textContent===name && document.querySelector('#detail').textContent.includes('batch '+batch)",arg=[name,batch])
        print('verified',name,batch,stage)
    page.screenshot(path=str(root/'Model/mae_adaptation_dashboard.png'))
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    browser.close()
assert not errors,errors
print('17_experiments_browser_verified_no_errors')
