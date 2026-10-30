"""Loopback-only viewer for verified triplets, hosted on the compute node."""
import argparse,functools,io,json,mimetypes,socket,threading,urllib.parse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from PIL import Image,ImageOps

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--port',type=int,default=55317);args=ap.parse_args()
    assert socket.gethostname()=='trinity-2-23'
    root=Path(args.root).resolve();base=root/'Dataset/Photo_Triplets_v1'
    records=json.loads((base/'triplets.json').read_text())['triplets'];byid={r['pair_id']:r for r in records}
    pilot={r['pair_id'] for r in json.loads((root/'Dataset/Photo_Pairs_v1/pilot_200.json').read_text())['pairs']}
    cache=root/'Dataset/Triplet_Review/preview_cache';cache.mkdir(parents=True,exist_ok=True)
    locks=[threading.Lock() for _ in range(64)]
    page=(Path(__file__).parent/'triplet_view.html').read_bytes()
    @functools.lru_cache(maxsize=32)
    def select(split,search):
        return [r for r in records if (split=='all' or split=='pilot' and r['pair_id'] in pilot or r['split']==split) and search in (' '.join([r['pair_id'],r['content_key'],r['reference_key']])).lower()]
    class Handler(BaseHTTPRequestHandler):
        def send(self,data,kind='application/json; charset=utf-8',status=200,cache_control='no-store'):
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control',cache_control);self.end_headers();self.wfile.write(data)
        def do_GET(self):
            try:
                url=urllib.parse.urlsplit(self.path);q=urllib.parse.parse_qs(url.query)
                if url.path in ['/', '/index.html']:return self.send(page,'text/html; charset=utf-8')
                if url.path=='/api/pairs':
                    selected=select(q.get('split',['pilot'])[0],q.get('search',[''])[0].strip().lower());index=max(0,min(int(q.get('index',['0'])[0]),max(0,len(selected)-1)))
                    wanted=q.get('pair',[''])[0]
                    if wanted:
                        index=next((i for i,r in enumerate(selected) if r['pair_id']==wanted),index)
                    return self.send(json.dumps({'total':len(selected),'index':index,'all_total':len(records),'items':selected[index:index+2]},ensure_ascii=False).encode())
                if url.path.startswith('/image/'):
                    _,_,ident,role=url.path.split('/')
                    assert role in ['content','reference','target']
                    r=byid[ident];source=(base/r[role+'_path']).resolve();source.relative_to(root/'Dataset')
                    if q.get('original',['0'])[0]=='1' or role=='target':
                        return self.send(source.read_bytes(),mimetypes.guess_type(source.name)[0] or 'application/octet-stream',cache_control='private, max-age=86400')
                    import hashlib
                    name=hashlib.sha256(str(source).encode()).hexdigest()+'.jpg';dest=cache/name
                    with locks[int(name[:8],16)%64]:
                        if not dest.exists():
                            with Image.open(source) as im:
                                im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((1024,1024),Image.Resampling.LANCZOS);im.save(dest,quality=94)
                    return self.send(dest.read_bytes(),'image/jpeg',cache_control='private, max-age=86400')
                self.send(b'Not found','text/plain',404)
            except (BrokenPipeError,ConnectionResetError):pass
            except Exception as e:self.send(json.dumps({'error':str(e)}).encode(),status=400)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Ready: {len(records)} triplets, http://127.0.0.1:{args.port}',flush=True);server.serve_forever()

if __name__=='__main__':main()
