"""Build an offline, self-contained pair browser without copying photographs."""
import json
from pathlib import Path
from urllib.parse import quote
import collect_photos as c

ROOT=c.ROOT/'Dataset'/'Photo_Pairs_v1'

def main():
    pairs=json.loads((ROOT/'pairs.json').read_text(encoding='utf-8'))['pairs']
    pilot={p['pair_id'] for p in json.loads((ROOT/'pilot_200.json').read_text(encoding='utf-8'))['pairs']}
    images=json.loads((ROOT/'source_images.json').read_text(encoding='utf-8'))['images']
    metadata={r['key']:{'source':r['source'],'width':r['width'],'height':r['height'],'title':r['title'],'artists':r['artists'],'preview':quote('../Original Data/_collection/'+r['preview'],safe='/')} for r in images}
    for p in pairs:
        p['pilot']=p['pair_id'] in pilot
        for field in ['content_path','reference_path']:
            assert (ROOT/p[field]).is_file()
            p[field]=quote(p[field],safe='/')
    template=(Path(__file__).parent/'pair_preview.html').read_text(encoding='utf-8')
    page=template.replace('__PAIRS__',json.dumps(pairs,ensure_ascii=False).replace('</','<\\/')).replace('__IMAGES__',json.dumps(metadata,ensure_ascii=False).replace('</','<\\/'))
    (ROOT/'preview.html').write_text(page,encoding='utf-8')
    print(json.dumps({'pairs':len(pairs),'pilot':len(pilot),'images':len(images),'page':str(ROOT/'preview.html')}))

if __name__=='__main__':main()
