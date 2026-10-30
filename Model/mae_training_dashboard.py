"""Read-only loopback dashboard, served on the compute node independently of SSH."""
import argparse
from functools import lru_cache
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
import statistics
from pathlib import Path
import time
from urllib.parse import unquote,urlparse,parse_qs
from mae_gallery import gallery_payload,safe_asset


def read_json(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def snapshot(root):
    runs=[];supervisor=read_json(root/'supervisor.json')
    supervisor['jobs']=list(supervisor.get('jobs',[]))
    for path in (root/'additional_supervisors').glob('*.json'):
        supervisor['jobs'].extend(read_json(path).get('jobs',[]))
    for folder in sorted((root/'runs').glob('*')):
        if not folder.is_dir():continue
        status=read_json(folder/'status.json');config=read_json(folder/'config.json')
        train=[];valid=[]
        path=folder/'metrics.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                try:event=json.loads(line)
                except ValueError:continue
                if event.get('resume'):
                    train=[point for point in train if point[0]<=event['step']]
                    valid=[point for point in valid if point[0]<=event['step']]
                for name,values in [('train',train),('validation',valid)]:
                    if name in event:
                        loss=event[name];values.append([event['step'],loss['total'],loss['reconstruction'],loss['structure'],event['elapsed_seconds']])
        alive=False
        if status.get('pid'):
            try:os.kill(status['pid'],0);alive=True
            except (OSError,ProcessLookupError):pass
        if status.get('state') in ('running','validating') and not alive:status['state']='interrupted'
        owner=next((job for job in supervisor.get('jobs',[]) if job['name']==folder.name),{})
        if owner.get('state')=='failed':status.update(state='failed',error=owner.get('error',status.get('error','Training process failed; inspect run log.')))
        eta=None
        if len(train)>2 and status.get('state') in ('running','validating'):
            recent=train[-15:];rates=[]
            for first,last in zip(recent,recent[1:]):
                if not any(first[0]<p[0]<=last[0] for p in valid) and last[0]>first[0]:
                    rates.append((last[4]-first[4])/(last[0]-first[0]))
            if rates:
                rate=statistics.median(rates)
                remaining=max(0,status.get('total_steps',0)-status.get('step',0))
                eta=remaining*rate
                if status.get('stage')=='heads':eta*=1.25
        gallery=read_json(folder/'gallery.json')
        gallery_summary={key:gallery.get(key) for key in ('version','checkpoint_step','checkpoint_kind','updated_unix','complete')}
        gallery_summary['ready']={split:len(gallery.get(split,[])) for split in ('train','validation')}
        runs.append({'name':folder.name,'status':status,'gallery':gallery_summary,'config':{k:config.get(k) for k in ('local_rank','seed','long_edge','accumulate','effective_batch','physical_batch_size','lora_rank','heads_steps','total_steps','train_count','validation_count','model')},
                     'train':train,'valid':valid,'alive':alive,'eta_seconds':eta})
    return {'time_unix':time.time(),'runs':runs,'supervisor':supervisor,'gallery_worker':read_json(root/'gallery_worker.json'),
            'note':'Train uses source-disjoint generated triplets; valid is fixed. No test metrics are used here.'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--port',type=int,default=55318)
    args=ap.parse_args();root=args.root.resolve();page=Path(__file__).with_name('mae_training_dashboard.html').read_bytes()
    gallery_js=Path(__file__).with_name('mae_gallery_ui.js').read_bytes()
    cache={'at':0.,'data':b''}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route=urlparse(self.path).path
            if route=='/':body=page;mime='text/html; charset=utf-8'
            elif route=='/gallery-ui.js':body=gallery_js;mime='text/javascript; charset=utf-8'
            elif route=='/api/gallery':
                query=parse_qs(urlparse(self.path).query)
                try:
                    payload=gallery_payload(root,query.get('run',[''])[0],query.get('split',[''])[0],
                                            int(query.get('page',['0'])[0]),int(query.get('size',['6'])[0]))
                except ValueError:self.send_error(400);return
                body=json.dumps(payload,separators=(',',':')).encode();mime='application/json'
            elif route.startswith('/gallery/'):
                try:path=safe_asset(root,unquote(route[len('/gallery/'):]))
                except ValueError:self.send_error(404);return
                body=path.read_bytes();mime={'.png':'image/png','.jpg':'image/jpeg','.webp':'image/webp'}[path.suffix.lower()]
            elif route=='/api/state':
                if time.time()-cache['at']>3:
                    cache['data']=json.dumps(snapshot(root),separators=(',',':'),allow_nan=False).encode();cache['at']=time.time()
                body=cache['data'];mime='application/json'
            elif route.startswith('/preview/'):
                run=unquote(route[len('/preview/'):]);path=(root/'runs'/run/'preview/latest.png').resolve()
                if '/' in run or '\\' in run or not path.is_relative_to(root/'runs') or not path.is_file():self.send_error(404);return
                body=path.read_bytes();mime='image/png'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    print('dashboard',f'http://127.0.0.1:{args.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__=='__main__':main()
