"""Integrated Qwen text proposals + optional Qwen-VL visual evidence.

Vision cooling: after approximately 5 minutes of active vision processing,
finish the current candidate, stop llama-server, rest 10 minutes, and restart.


Examples:
 python -m scripts.generate_qwen_candidates --source-file 1 --page 4 --vision --dry-run
 python -m scripts.generate_qwen_candidates --source-file 1 --page 4 --vision --llama-exe C:\\Tools\\llama.cpp\\llama-server.exe

Requires the qwen_vision_evidence table defined in sql/schema.sql for vision database writes.
No separate SQL migration is required.
"""
from __future__ import annotations
import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import request, error
from src.database import get_connection
from src.repository import get_tokens_for_contextual_review, upsert_correction_candidate
from src.qwen_correction import QwenCorrectionModel, PROMPT_VERSION
from src.image_crops import fetch_token_geometry, make_crop
from src.qwen_vision import transcribe_crop

ROOT = Path(__file__).resolve().parents[1]
SUBSTITUTION = re.compile(r"['\"]([^'\"]+)['\"]\s+(?:instead of|replaced by|rather than)\s+['\"]([^'\"]+)['\"]", re.I)

def group_by_page(rows):
    pages = {}
    for token_id, raw_token, token_type, page_id, source_file_id, source_page_number, corpus_page_number, position_in_page in rows:
        page = pages.setdefault(page_id, {'page_id':page_id,'source_file_id':source_file_id,'source_page_number':source_page_number,'corpus_page_number':corpus_page_number,'tokens':[]})
        page['tokens'].append({'token_id':token_id,'raw_token':raw_token,'token_type':token_type,'position_in_page':position_in_page})
    for page in pages.values():
        page['tokens'].sort(key=lambda t:t['position_in_page'])
    return sorted(pages.values(), key=lambda p:(p['corpus_page_number'],p['page_id']))

def validate(item, tokens, seen):
    if not isinstance(item, dict): return None, 'not an object'
    ident = item.get('token_id')
    if isinstance(ident, bool): return None, 'boolean token ID'
    try: ident = int(ident)
    except (TypeError, ValueError): return None, 'invalid token ID'
    if ident not in tokens: return None, 'token outside page'
    if ident in seen: return None, 'duplicate token'
    raw, suggestion, reason = item.get('raw_token'), item.get('suggested_token'), item.get('reason')
    if raw != tokens[ident]['raw_token']: return None, 'token text mismatch'
    if not isinstance(suggestion,str) or not suggestion.strip(): return None, 'no suggestion'
    suggestion = suggestion.strip()
    if suggestion == raw: return None, 'unchanged token'
    if any(c.isspace() for c in suggestion): return None, 'multi-token proposal'
    if not isinstance(reason,str) or not reason.strip(): return None, 'no reason'
    match = SUBSTITUTION.search(reason)
    if match and match.group(1) == match.group(2): return None, 'contradictory reason'
    seen.add(ident)
    return {'token_id':ident,'raw_token':raw,'suggested_token':suggestion,'reason':reason.strip()}, None

def audit(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a',encoding='utf-8') as f: f.write(json.dumps(record,ensure_ascii=False,default=str)+'\n')

def server_ready(url, model):
    try:
        with request.urlopen(url.rstrip('/')+'/v1/models',timeout=4) as f: data=json.load(f)
        return any(m.get('id') == model for m in data.get('data',[]))
    except (OSError, ValueError, error.URLError): return False

class ManagedServer:
    def __init__(self, args):
        self.args=args; self.process=None
    def start(self, model_path, alias, mmproj=None, context=8192):
        if server_ready(self.args.server_url,alias):
            raise RuntimeError('An existing server is listening. Stop it first; automated switching requires exclusive control of port 8080.')
        exe = Path(self.args.llama_exe)
        if not exe.is_file(): raise FileNotFoundError(exe)
        model_path = Path(model_path).resolve()
        if not model_path.is_file(): raise FileNotFoundError(model_path)
        cmd=[str(exe),'-m',str(model_path),'-ngl','99','-c',str(context),'--parallel','1','--host','127.0.0.1','--port',str(self.args.port),'--alias',alias]
        if mmproj:
            projector=Path(mmproj).resolve()
            if not projector.is_file(): raise FileNotFoundError(projector)
            cmd += ['--mmproj',str(projector),'--image-min-tokens','1024']
        log_dir=ROOT/'data'/'experiments'/'qwen_integrated'
        log_dir.mkdir(parents=True,exist_ok=True)
        log_path=log_dir/f'server_{alias}_{int(time.time())}.log'
        self.log=open(log_path,'w',encoding='utf-8')
        self.process=subprocess.Popen(cmd,stdout=self.log,stderr=subprocess.STDOUT,cwd=str(ROOT))
        print(f'Starting {alias}; server log: {log_path}',flush=True)
        for _ in range(180):
            if self.process.poll() is not None: raise RuntimeError(f'{alias} server exited early; inspect {log_path}')
            if server_ready(self.args.server_url,alias): return
            time.sleep(2)
        self.stop()
        raise TimeoutError(f'{alias} did not become ready')
    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try: self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.kill(); self.process.wait()
            self.process=None
        if getattr(self,'log',None): self.log.close(); self.log=None
        time.sleep(2)

def classify(transcription, suggestion, raw):
    # Conservative machine agreement, not acceptance: whole-token boundaries.
    words = re.findall(r"[^\W_]+(?:['’-][^\W_]+)*",transcription,flags=re.UNICODE)
    proposed = suggestion.casefold()
    original = raw.casefold()
    matches_proposal = proposed in [w.casefold() for w in words]
    matches_original = original in [w.casefold() for w in words]
    if matches_proposal and not matches_original: return 'SUPPORT'
    if matches_original and not matches_proposal: return 'CONTRADICT'
    return 'UNCERTAIN'

def saved_qwen_matches(token_id, suggestion):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""SELECT 1 FROM correction_candidate cc JOIN token t ON t.token_id=cc.token_id
                WHERE cc.token_id=%s AND cc.method='qwen' AND cc.suggested_token=%s
                AND t.correction_reviewed=FALSE""", (token_id, suggestion))
            return cur.fetchone() is not None


def has_vision_evidence(token_id, suggestion):
    """UNCERTAIN is completed evidence too; skip all recorded decisions."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""SELECT 1 FROM correction_candidate cc
                JOIN token t ON t.token_id=cc.token_id
                JOIN qwen_vision_evidence v ON v.candidate_id=cc.candidate_id
                WHERE cc.token_id=%s AND cc.method='qwen' AND cc.suggested_token=%s
                AND t.correction_reviewed=FALSE AND v.model_name='qwen3-vl-8b'
                AND v.model_quantization='Q4_K_M'""", (token_id, suggestion))
            return cur.fetchone() is not None


def load_saved_qwen_jobs(source_file_id, page_number, max_pages):
    """Load persisted, unreviewed text proposals in physical page order."""
    filters = ["cc.method='qwen'", "t.correction_reviewed=FALSE", "cc.suggested_token IS NOT NULL"]
    params = []
    if source_file_id is not None:
        filters.append('p.source_file_id=%s'); params.append(source_file_id)
    if page_number is not None:
        filters.append('p.source_page_number=%s'); params.append(page_number)
    sql = """SELECT t.token_id, t.raw_token, cc.suggested_token, p.page_id,
        p.source_file_id, p.source_page_number, p.corpus_page_number
        FROM correction_candidate cc JOIN token t ON t.token_id=cc.token_id
        JOIN ocr_region r ON r.region_id=t.region_id
        JOIN ocr_run o ON o.ocr_run_id=r.ocr_run_id
        JOIN page p ON p.page_id=o.page_id
        WHERE """ + ' AND '.join(filters) + ' ORDER BY p.corpus_page_number,t.position_in_page'
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    jobs = []
    selected_pages = set()
    for token_id, raw, suggestion, page_id, source_id, source_page, corpus_page in rows:
        if page_id not in selected_pages and max_pages is not None and len(selected_pages) >= max_pages:
            break
        selected_pages.add(page_id)
        page = {'page_id':page_id,'source_file_id':source_id,
                'source_page_number':source_page,'corpus_page_number':corpus_page}
        item = {'token_id':token_id,'raw_token':raw,'suggested_token':suggestion}
        jobs.append((page,item))
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-file',type=int)
    p.add_argument('--page',type=int)
    p.add_argument('--max-pages',type=int)
    p.add_argument('--vision',action='store_true')
    p.add_argument('--vision-only',action='store_true',help='Verify saved Qwen proposals without rerunning text inference')
    p.add_argument('--resume',action='store_true',help='Skip completed vision evidence (default behavior)')
    p.add_argument('--force',action='store_true',help='Re-run completed vision verification; never alter reviewed tokens')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--verbose',action='store_true')
    p.add_argument('--server-url',default='http://127.0.0.1:8080')
    p.add_argument('--port',type=int,default=8080)
    p.add_argument('--llama-exe',default=r'C:\Tools\llama.cpp\llama-server.exe')
    p.add_argument('--text-gguf',default=str(ROOT/'models/qwen3-8b-q6/Qwen3-8B-Q6_K.gguf'))
    p.add_argument('--vision-gguf',default=str(ROOT/'models/qwen3-vl-8b/Qwen3VL-8B-Instruct-Q4_K_M.gguf'))
    p.add_argument('--mmproj',default=str(ROOT/'models/qwen3-vl-8b/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf'))
    p.add_argument('--padding-x',type=int,default=650)
    p.add_argument('--padding-y',type=int,default=100)
    p.add_argument('--context-size',type=int,default=12288)
    p.add_argument('--max-new-tokens',type=int,default=4096)
    p.add_argument('--vision-work-minutes',type=float,default=5.0,help='Active vision processing time before cooling (default: 5)')
    p.add_argument('--vision-rest-minutes',type=float,default=10.0,help='Cooling duration with vision server stopped (default: 10)')
    args=p.parse_args()
    if args.page is not None and args.source_file is None: p.error('--page requires --source-file')
    if args.vision_only: args.vision = True
    if args.force and not args.vision: p.error('--force requires --vision or --vision-only')
    if args.max_pages is not None and args.max_pages < 1: p.error('--max-pages must be positive')
    if min(args.padding_x,args.padding_y)<0: p.error('padding must be nonnegative')
    if args.vision_work_minutes <= 0: p.error('--vision-work-minutes must be positive')
    if args.vision_rest_minutes < 0: p.error('--vision-rest-minutes cannot be negative')
    if args.port != 8080: args.server_url=f'http://127.0.0.1:{args.port}'
    # Vision-only retrieves candidates from PostgreSQL, not the prior process's jobs list.
    if args.vision_only:
        pages = []
    else:
        with get_connection() as conn:
            rows = get_tokens_for_contextual_review(conn, source_file_id=args.source_file)
        pages = group_by_page(rows)
        if args.page is not None:
            pages = [x for x in pages if x['source_page_number'] == args.page]
        if args.max_pages:
            pages = pages[:args.max_pages]
        if not pages:
            print('No eligible tokens.'); return

    print(f'Pages: {len(pages)} | vision: {args.vision} | vision-only: {args.vision_only} | dry-run: {args.dry_run}')
    if args.vision:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass('public.qwen_vision_evidence')")
                if cur.fetchone()[0] is None:
                    raise RuntimeError('Missing public.qwen_vision_evidence. Create it in sql/schema.sql.')
    manager=ManagedServer(args)
    jobs=[]; failures=0
    try:
        if not args.vision_only:
            manager.start(args.text_gguf,'qwen3-8b-q6',context=args.context_size)
            model=QwenCorrectionModel(model_name='qwen3-8b-q6',server_url=args.server_url,context_size=args.context_size,max_new_tokens=args.max_new_tokens,verbose=args.verbose)
            for page in pages:
                page_no=page['source_page_number']; started=time.perf_counter()
                print(f'TEXT page {page_no:04d} ({len(page["tokens"])} tokens)',flush=True)
                record={'stage':'text','timestamp':datetime.now(timezone.utc).isoformat(),'page':page_no,'source_file_id':page['source_file_id'],'model':'qwen3-8b-q6','prompt_version':PROMPT_VERSION,'dry_run':args.dry_run}
                try:
                    proposals=model.analyze_page(page['tokens'])
                    token_map={t['token_id']:t for t in page['tokens']}; seen=set(); valid=[]; rejected=[]
                    for proposal in proposals:
                        item,reason=validate(proposal,token_map,seen)
                        if reason: rejected.append({'item':proposal,'reason':reason})
                        else: valid.append(item)
                    record.update({'raw_response':model.last_response,'valid':valid,'rejected':rejected,'status':'ok'})
                    if not args.dry_run:
                        with get_connection() as conn:
                            for item in valid:
                                # Do not overwrite any existing reviewed token, and avoid upserting prior qwen proposals.
                                with conn.cursor() as cur:
                                    cur.execute('SELECT correction_reviewed FROM token WHERE token_id=%s',(item['token_id'],))
                                    if cur.fetchone()[0]: continue
                                    cur.execute("SELECT candidate_id FROM correction_candidate WHERE token_id=%s AND method='qwen'",(item['token_id'],))
                                    if cur.fetchone():
                                        print(f'  SKIP existing qwen candidate {item["token_id"]}; no overwrite')
                                        continue
                                upsert_correction_candidate(conn,token_id=item['token_id'],method='qwen',suggested_token=item['suggested_token'],details={'reason':item['reason'],'model':'qwen3-8b-q6','quantization':'Q6_K','prompt_version':PROMPT_VERSION})
                            conn.commit()
                    for item in valid:
                        if args.dry_run or saved_qwen_matches(item['token_id'], item['suggested_token']):
                            jobs.append((page,item))
                        else:
                            print(f"  SKIP vision [{item['token_id']}]: proposal differs from stored Qwen candidate", flush=True)
                    print(f'  Valid: {len(valid)} | rejected: {len(rejected)}',flush=True)
                except Exception as exc:
                    failures+=1; record.update({'status':'failed','error':str(exc),'raw_response':model.last_response})
                    print(f'  TEXT FAILED: {exc}',flush=True)
                finally:
                    record['elapsed_seconds']=round(time.perf_counter()-started,2)
                    audit(ROOT/'data'/'experiments'/'qwen_integrated'/f'source_{page["source_file_id"]}_page_{page_no:04d}.jsonl',record)
            manager.stop()
        if args.vision_only:
            jobs = load_saved_qwen_jobs(args.source_file, args.page, args.max_pages)
            print(f'Loaded {len(jobs)} saved, unreviewed Qwen proposals for vision verification.', flush=True)
        if args.vision and jobs:
            if not args.force:
                jobs = [(page, item) for page, item in jobs if not has_vision_evidence(item['token_id'], item['suggested_token'])]
                print(f'Pending vision candidates: {len(jobs)}', flush=True)
        if args.vision and jobs:
            manager.start(args.vision_gguf,'qwen3-vl-8b',mmproj=args.mmproj,context=4096)
            vision_work_started = time.monotonic()
            vision_work_seconds = args.vision_work_minutes * 60
            vision_rest_seconds = args.vision_rest_minutes * 60
            for job_index, (page,item) in enumerate(jobs):
                # Only rest between candidates, never during an inference or DB write.
                # No cooling delay after the last candidate.
                if job_index > 0 and time.monotonic() - vision_work_started >= vision_work_seconds:
                    print(f'\n[COOLING] Vision active for at least {args.vision_work_minutes:g} min; stopping GPU server.',flush=True)
                    manager.stop()
                    if vision_rest_seconds:
                        print(f'[COOLING] Resting {args.vision_rest_minutes:g} min; {len(jobs)-job_index} candidates remain.',flush=True)
                        rest_end = time.monotonic() + vision_rest_seconds
                        while True:
                            remaining = rest_end - time.monotonic()
                            if remaining <= 0:
                                break
                            time.sleep(min(30, remaining))
                    print('[COOLING] Restarting Qwen-VL server.',flush=True)
                    manager.start(args.vision_gguf,'qwen3-vl-8b',mmproj=args.mmproj,context=4096)
                    vision_work_started = time.monotonic()
                token_id=item['token_id']; page_no=page['source_page_number']
                record={'stage':'vision','timestamp':datetime.now(timezone.utc).isoformat(),'token_id':token_id,'page':page_no,'proposal':item,'dry_run':args.dry_run}
                try:
                    with get_connection() as conn: geom=fetch_token_geometry(conn,token_id)
                    if geom['raw_token']!=item['raw_token']: raise RuntimeError('Database token changed since text stage')
                    crop,box=make_crop(geom,padding_x=args.padding_x,padding_y=args.padding_y)
                    path=ROOT/'data'/'experiments'/'vision_crops'/f'token_{token_id}_page_{page_no:04d}.png'
                    path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(crop)
                    result=transcribe_crop(crop,url=args.server_url)
                    decision=classify(result['transcription'],item['suggested_token'],item['raw_token'])
                    record.update({'status':'ok','vision':result,'decision':decision,'crop_box':box,'crop_path':str(path)})
                    print(f'VISION [{token_id}] {item["raw_token"]!r} -> {item["suggested_token"]!r}: {decision} | {result["transcription"][:120]!r}',flush=True)
                    if not args.dry_run:
                        with get_connection() as conn:
                            with conn.cursor() as cur:
                                cur.execute("SELECT cc.candidate_id FROM correction_candidate cc JOIN token t ON t.token_id=cc.token_id WHERE cc.token_id=%s AND cc.method='qwen' AND cc.suggested_token=%s AND t.correction_reviewed=FALSE",(token_id,item['suggested_token']))
                                row=cur.fetchone()
                                if row is None:
                                    print('  SKIP vision DB write: matching unreviewed candidate not found')
                                else:
                                    cur.execute('''INSERT INTO qwen_vision_evidence(candidate_id,model_name,model_quantization,transcription,decision,crop_path,crop_box,metadata)
                                        VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
                                        ON CONFLICT(candidate_id,model_name,model_quantization) DO UPDATE SET transcription=EXCLUDED.transcription,decision=EXCLUDED.decision,crop_path=EXCLUDED.crop_path,crop_box=EXCLUDED.crop_box,metadata=EXCLUDED.metadata,created_at=CURRENT_TIMESTAMP''',
                                        (row[0],'qwen3-vl-8b','Q4_K_M',result['transcription'],decision,str(path.relative_to(ROOT)).replace('\\','/'),json.dumps(box),json.dumps({'usage':result.get('usage'),'verification_type':'neutral_image_transcription'})))
                            conn.commit()
                except Exception as exc:
                    failures+=1; record.update({'status':'failed','error':str(exc)})
                    print(f'  VISION FAILED [{token_id}]: {exc}',flush=True)
                finally:
                    audit(ROOT/'data'/'experiments'/'qwen_integrated'/f'source_{page["source_file_id"]}_page_{page_no:04d}.jsonl',record)
    finally:
        manager.stop()
    print(f'Complete: {len(jobs)} text proposals; failures={failures}; human review still required.')
    if failures: raise SystemExit(1)

if __name__=='__main__': main()
