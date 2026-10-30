"""Browser verification against the live SSH-forwarded gallery."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

root=Path(__file__).resolve().parents[1]
errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1100})
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto('http://127.0.0.1:55318/',wait_until='networkidle')
    page.wait_for_selector('#validationGallery .sample-row',timeout=60000)
    page.wait_for_selector('#trainGallery .sample-row',timeout=60000)
    assert page.locator('.run').count()==9
    assert page.locator('#validationGallery').bounding_box()['y'] < page.locator('#trainGallery').bounding_box()['y']
    for split in ('validation','train'):
        assert page.locator('#'+split+'Gallery .sample-row').count()==6
        first=page.locator('#'+split+'Gallery .sample-label').first.inner_text()
        page.locator('#'+split+'Next').click()
        page.wait_for_function("([split,first])=>document.querySelector('#'+split+'Gallery .sample-label').innerText!==first",arg=[split,first])
        page.locator('#'+split+'Prev').click()
    page.locator('#validationGallery .gallery-image').first.click()
    page.wait_for_function("document.getElementById('galleryZoom').open && document.getElementById('zoomImage').naturalWidth>0")
    page.screenshot(path=str(root/'Model/mae_gallery_zoom.png'))
    page.keyboard.press('Escape')
    assert not page.locator('#galleryZoom').evaluate('(d)=>d.open')
    page.locator('#validationGallery').scroll_into_view_if_needed()
    page.screenshot(path=str(root/'Model/mae_gallery_desktop.png'))
    page.select_option('#runSelect','M1_LoRA_r8_b16_s17')
    page.wait_for_function("document.querySelector('#trainGallery .sample-row') && document.querySelector('#trainGalleryInfo').textContent.includes('48/48')",timeout=60000)
    page.set_viewport_size({'width':390,'height':844})
    page.locator('#trainGallery .sample-row').first.scroll_into_view_if_needed()
    page.wait_for_function("[...document.querySelectorAll('#trainGallery .sample-row:first-child img')].every(i=>i.complete&&i.naturalWidth>0)")
    page.screenshot(path=str(root/'Model/mae_gallery_mobile.png'))
    assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth'), 'mobile horizontal overflow'
    print('nine_runs_two_galleries_pagination_zoom_mobile_verified',errors)
    browser.close()
if errors:raise RuntimeError(errors)
