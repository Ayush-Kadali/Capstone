"""Render the current capstone guides as one offline, searchable HTML book.

Run: python3 scripts/build_html_guides.py
Build dependencies: markdown-it-py, beautifulsoup4 (reading needs only a browser).
Markdown remains authoritative. No remote fonts, scripts, or analytics.
"""
from pathlib import Path
from html import escape
import hashlib
import json
import re
from urllib.parse import urlsplit, unquote
from markdown_it import MarkdownIt
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'showcase/html-guide'
DOCS = [
    ('cli', 'Developer CLI', 'docs/runtime/DEVELOPER_CLI.md'),
    ('onnx', 'ONNX model input', 'docs/runtime/ONNX_FRONTEND.md'),
    ('targets', 'Simulator target profiles', 'docs/runtime/TARGET_PROFILES.md'),
    ('analytical', 'Analytical performance model', 'docs/runtime/ANALYTICAL_MODEL.md'),
    ('trained', 'Trained example', 'docs/runtime/TRAINED_EXAMPLE.md'),
    ('learn', 'Learning guide', 'showcase/ENGINEERING_NOTES.md'),
    ('demo', 'Demo & reproduction', 'docs/runtime/DEMONSTRATION.md'),
    ('architecture', 'Architecture', 'docs/runtime/ARCHITECTURE.md'),
    ('contracts', 'Device contracts', 'docs/runtime/DEVICE_CONTRACT.md'),
    ('memory', 'Memory & lifetimes', 'docs/runtime/MEMORY.md'),
    ('residency', 'Input residency', 'docs/runtime/RESIDENCY.md'),
    ('kernels', 'Kernels & numerics', 'docs/runtime/KERNELS.md'),
    ('async', 'Queues & completion', 'docs/runtime/ASYNC.md'),
    ('queue', 'Command-level queue', 'docs/runtime/COMMAND_QUEUE.md'),
    ('bounded', 'Larger model placement', 'docs/runtime/BOUNDED_SEARCH.md'),
    ('study', 'Placement experiments', 'docs/runtime/STUDY.md'),
    ('production', 'Production comparison', 'docs/runtime/PRODUCTION_COMPARISON.md'),
    ('performance', 'Performance investigation', 'docs/runtime/PERFORMANCE_INVESTIGATION.md'),
    ('debugging', 'Debugging dossier', 'docs/runtime/DEBUGGING_DOSSIER.md'),
    ('workbench', 'Workbench instructions', 'workbench/README.md'),
    ('validation', 'Validation evidence', 'docs/runtime/VALIDATION.md'),
    ('acceptance', 'Release acceptance record', 'docs/runtime/COLLEGE_RELEASE_ACCEPTANCE.md'),
    ('hardware', 'Hardware roadmap (deferred)', 'planning/HARDWARE_ROADMAP.md'),
    ('completion', 'Completion matrix (historical)', 'docs/runtime/COMPLETION_MATRIX.md'),
    ('ti', 'TI adapter preparation', 'docs/runtime/TI_ADAPTER.md'),
    ('board', 'Board validation', 'docs/runtime/BOARD_VALIDATION.md'),
    ('handoff', 'Board handoff', 'docs/runtime/BOARD_HANDOFF.md'),
]
MD = MarkdownIt('commonmark', {'html': False}).enable('table')


def slug(value):
    return re.sub(r'[^\w\- ]', '', value.lower()).replace(' ', '-')


CSS = '''
:root{--ink:#172c40;--muted:#4d6171;--paper:#f5f4ef;--surface:#fff;--line:#ced8dc;--accent:#075d60;--tint:#e6f1ee;--code:#102936}
*{box-sizing:border-box}html{scroll-padding-top:30px}body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:var(--accent);text-underline-offset:3px}button,input{font:inherit}a:focus-visible,button:focus-visible,input:focus-visible,summary:focus-visible{outline:3px solid #ba6518;outline-offset:4px}button,summary{cursor:pointer}button,.action{padding:10px 16px;border:1px solid var(--accent);background:var(--surface);color:var(--accent);border-radius:5px;text-decoration:none;display:inline-block;min-height:44px}button:hover,.action:hover{background:var(--tint)}.primary{background:var(--accent);color:white}.primary:hover{background:#08494b}.skip{position:fixed;top:-100px;left:12px;z-index:10;background:white;padding:12px}.skip:focus{top:8px}
.layout{display:grid;grid-template-columns:285px minmax(0,1fr);min-height:100vh}aside{position:sticky;top:0;height:100vh;overflow:auto;background:#eaf0ef;border-right:1px solid var(--line);padding:30px 20px}.eyebrow{font-size:12px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted);font-weight:700}.brand{font-size:23px;line-height:1.2;margin:10px 0 24px}.brand a{text-decoration:none;color:var(--ink)}label{display:block;font-size:14px;font-weight:600}input{width:100%;padding:10px;background:white;border:1px solid #84999d;border-radius:4px;margin:5px 0 10px}nav a{display:block;text-decoration:none;padding:8px 10px;min-height:44px;border-left:3px solid transparent;font-size:14px;color:var(--ink)}nav a:hover{background:#dce7e4}nav a[aria-current=page]{border-color:var(--accent);background:white;font-weight:700}.small{font-size:13px;color:var(--muted)}#results{font-size:14px}#results a{display:block;padding:8px 0;border-bottom:1px solid var(--line)}
main{min-width:0;padding:45px clamp(22px,5vw,78px) 80px;max-width:1250px}.top{display:flex;gap:15px;justify-content:space-between;align-items:center;margin-bottom:35px}.top p{margin:0;font-size:13px;color:var(--muted)}h1,h2,h3{line-height:1.22;text-wrap:balance}h1{font-size:clamp(32px,4vw,54px);letter-spacing:-.035em;margin:15px 0 24px}h2{font-size:28px;margin:52px 0 16px}h3{font-size:21px;margin:32px 0 14px}p,li{max-width:78ch}p{margin:0 0 20px}li{margin:5px 0}.lead{font-size:21px;line-height:1.55;color:var(--muted);max-width:64ch}.notice{border-left:4px solid #a76b1b;padding:14px 20px;background:#fff4de;margin:28px 0;font-size:15px}.actions{display:flex;flex-wrap:wrap;gap:12px;margin:24px 0}.route{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;margin:35px 0}.route>a{display:block;background:white;border:1px solid var(--line);padding:23px;text-decoration:none;color:var(--ink)}.route strong{display:block;font-size:21px;margin:9px 0}.route span{font-size:14px;color:var(--muted)}.pipeline{display:flex;flex-wrap:wrap;gap:8px;margin:26px 0}.pipeline span{padding:7px 12px;background:var(--tint);border:1px solid #accbc4;font:14px ui-monospace,Menlo,monospace}
pre{padding:20px;background:var(--code);color:#edf5f7;overflow:auto;border-radius:6px;font-size:13px;line-height:1.65}code{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.85em;overflow-wrap:anywhere}p code,li code,td code{background:#e5ecec;padding:2px 4px}pre code{font-size:inherit;overflow-wrap:normal}blockquote{border-left:3px solid var(--accent);padding-left:20px;margin-left:0;color:var(--muted)}.table-wrap{overflow:auto;margin:24px 0;max-width:100%}table{border-collapse:collapse;width:100%;font-size:14px;background:white}td,th{padding:12px;border:1px solid var(--line);text-align:left;vertical-align:top;min-width:100px}th{background:#e7efed}details{margin:22px 0;border:1px solid var(--line);padding:15px;background:white}summary{font-weight:600;min-height:32px}.toc a{display:block;padding:5px 0;font-size:14px}.toc .sub{padding-left:20px}.source{font-size:12px;color:var(--muted);overflow-wrap:anywhere}.page[hidden]{display:none}.page{animation:none}footer{border-top:1px solid var(--line);margin-top:45px;padding-top:20px;font-size:13px;color:var(--muted)}.stage{padding:22px;border:1px solid var(--line);background:white;margin-top:18px}.step-controls{display:flex;justify-content:space-between;gap:15px}.result{font-size:24px;font-weight:700;color:var(--accent)}.answer{margin-top:25px}noscript{display:block;padding:20px}
@media(max-width:800px){.layout{display:block}aside{position:static;height:auto;padding:20px}nav{display:grid;grid-template-columns:repeat(2,minmax(0,1fr))}main{padding:25px 20px 60px}.route{grid-template-columns:1fr}.top{align-items:flex-start}h1{font-size:34px}}
@media print{aside,.top,.actions,.step-controls{display:none!important}.layout{display:block}main{padding:0;max-width:none}.page[hidden]{display:none}.page{display:block}pre{white-space:pre-wrap;background:#eee;color:#111}.table-wrap{overflow:visible}table{font-size:10px}h2,h3{break-after:avoid}a{color:#111}}
'''

JS = '''
const pages=[...document.querySelectorAll('.page')];
function show(){let target=document.getElementById(decodeURIComponent(location.hash.slice(1)));let page=target?.closest('.page')||document.getElementById('start');pages.forEach(p=>p.hidden=p!==page);document.querySelectorAll('nav a').forEach(a=>{if(a.hash==='#'+page.id)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current')});document.title=(page.dataset.title||'Start here')+' · Capstone guide';if(target&&target!==page){if(target.closest('details'))target.closest('details').open=true;target.scrollIntoView()}else if(innerWidth<=800)page.scrollIntoView();else window.scrollTo(0,0)}
window.addEventListener('hashchange',show);show();
const search=document.getElementById('search'), results=document.getElementById('results');
const entries=pages.flatMap(p=>[...p.querySelectorAll('h1,h2,h3')].map(h=>{let parts=[h.textContent],el=h.nextElementSibling;while(el&&!/^H[123]$/.test(el.tagName)){parts.push(el.textContent);el=el.nextElementSibling}return{id:h.id||p.id,title:h.textContent,page:p.dataset.title||'Start here',text:parts.join(' ').toLowerCase()}}));
search.addEventListener('input',()=>{results.replaceChildren();const q=search.value.trim().toLowerCase();if(!q)return;const matches=entries.filter(e=>e.text.includes(q));let label=document.createElement('p');label.textContent=matches.length+' matching sections'+(matches.length>30?' (first 30 shown)':'');results.append(label);matches.slice(0,30).forEach(e=>{const a=document.createElement('a');a.href='#'+e.id;a.textContent=e.page+' / '+e.title;results.append(a)})});
document.getElementById('print').addEventListener('click',()=>window.print());
let step=0;const stages=[...document.querySelectorAll('.stage')];function reveal(){stages.forEach((s,i)=>s.hidden=i!==step);document.getElementById('step-count').textContent='Step '+(step+1)+' of '+stages.length;document.getElementById('previous').disabled=step===0;document.getElementById('next').disabled=step===stages.length-1}document.getElementById('previous').onclick=()=>{step--;reveal()};document.getElementById('next').onclick=()=>{step++;reveal()};reveal();
'''


def write_reading_edition(output):
    """One transferable file: reading chapters and escaped source appendices."""
    soup = BeautifulSoup(output, 'html.parser')
    for key in ('tour', 'demo'):
        soup.find(id=key).decompose()
    for link in list(soup.select('nav a[href="#tour"], nav a[href="#demo"]')):
        link.decompose()
    soup.select_one('#start .actions').clear()
    action = soup.new_tag('a', href='#learn', attrs={'class': 'action primary'})
    action.string = 'Read the learning guide'
    soup.select_one('#start .actions').append(action)
    route = soup.select_one('#start .route a[href="#demo"]')
    route['href'] = '#architecture'
    route.clear()
    route.append(BeautifulSoup('<span>02 / Explore</span><strong>Trace the architecture</strong><span>Follow model import, placement, memory, kernels and runtime ownership.</span>', 'html.parser'))
    intro = soup.select_one('#start').find_all('p')[-1]
    intro.string = 'This is a single-file reading edition. All chapters, styles, search and linked text snapshots are embedded. Send this HTML file and open it in a browser; no server or companion folder is needed. External reference websites need internet access. The live workbench and executable demo are not included.'
    soup.title.string = 'Capstone guide · Single-file reading edition'
    soup.select_one('aside .small').string = 'Single-file reading edition · Works offline'
    soup.select_one('footer').string = 'Self-contained reading edition generated from project documentation. Embedded source snapshots are literal text, never executed. External websites are optional online references.'
    # Reading navigation/search remains useful; omit the removed demo controller.
    soup.find('script').string = JS.split('let step=0;')[0]
    main = soup.find('main')
    embedded = {}
    for link in list(soup.select('a[href]')):
        href = link['href']
        u = urlsplit(href)
        if href.startswith('sources/'):
            source = OUT / href
            data = source.read_bytes()
            try:
                text = data.decode('utf-8')
                if '\0' in text:
                    raise UnicodeError('binary content')
            except UnicodeError:
                link.replace_with(link.get_text() + ' (binary artifact; available in the project repository)')
                continue
            key = 'embedded-' + hashlib.sha256(href.encode()).hexdigest()[:16]
            if key not in embedded:
                title = link.get('title', source.name).removeprefix('Source snapshot: ')
                article = soup.new_tag('article', attrs={'class': 'page', 'id': key, 'data-title': title})
                label = soup.new_tag('p', attrs={'class': 'eyebrow'})
                label.string = 'Embedded source snapshot'
                article.append(label)
                heading = soup.new_tag('h1')
                heading.string = title
                article.append(heading)
                digest = soup.new_tag('p', attrs={'class': 'source'})
                digest.string = 'SHA-256 ' + hashlib.sha256(data).hexdigest()
                article.append(digest)
                back = soup.new_tag('a', href='#start')
                back.string = 'Back to the guide'
                article.append(back)
                pre = soup.new_tag('pre')
                code = soup.new_tag('code')
                code.string = text
                pre.append(code)
                article.append(pre)
                main.find('footer').insert_before(article)
                embedded[key] = hashlib.sha256(data).hexdigest()
            link['href'] = '#' + key
            link.attrs.pop('target', None)
        elif u.hostname in ('localhost', '127.0.0.1'):
            link.replace_with(link.get_text() + ' (local application; not included in this reading edition)')
        elif href == '#demo' or href.startswith('#demo--') or href == '#tour':
            link.replace_with(link.get_text() + ' (execution walkthrough omitted from reading edition)')
        elif u.scheme in ('https', 'http'):
            link['title'] = 'Online reference; internet required'
    # No relative-file dependencies or broken chapter/appendix links may escape.
    ids = {tag['id'] for tag in soup.select('[id]')}
    for link in soup.select('a[href]'):
        href = link['href']
        if href.startswith('#'):
            assert unquote(href[1:]) in ids, href
        else:
            assert urlsplit(href).scheme in ('https', 'http', 'mailto'), href
    assert not soup.select('script[src], link[rel="stylesheet"], img[src], iframe, object')
    destination = OUT / 'Capstone-Reading-Guide.html'
    destination.write_text(str(soup))
    print(f'{destination}: {len(DOCS)-1} reading chapters, {len(embedded)} embedded text snapshots, {destination.stat().st_size:,} bytes')


def main():
    OUT.mkdir(exist_ok=True)
    (OUT/'sources').mkdir(exist_ok=True)
    for stale in (OUT/'sources').glob('*'):  # snapshots are regenerated; drop orphans of renamed/removed links
        stale.unlink()
    mapping = {(ROOT/path).resolve(): key for key, _, path in DOCS}
    nav = '<a href="#start">Start here</a><a href="#tour">Five-minute demo</a>'
    pages = []
    manifest = {}
    for key, title, source in DOCS:
        path = ROOT/source
        raw = path.read_text()
        manifest[source] = hashlib.sha256(raw.encode()).hexdigest()
        soup = BeautifulSoup(MD.render(raw), 'html.parser')
        toc=[]; used={}
        for h in soup.find_all(re.compile('^h[1-6]$')):
            base=slug(h.get_text()); used[base]=used.get(base,0)+1
            h['id']=key+'--'+base+('' if used[base]==1 else '-'+str(used[base]-1))
            if h.name in ('h2','h3'):
                toc.append(f'<a class="{"sub" if h.name=="h3" else ""}" href="#{h["id"]}">{escape(h.get_text())}</a>')
        for a in soup.find_all('a',href=True):
            u=urlsplit(a['href'])
            if u.scheme or u.netloc:
                continue
            dest=(path.parent/unquote(u.path)).resolve() if u.path else path.resolve()
            if dest in mapping:
                a['href']='#'+mapping[dest]+('--'+unquote(u.fragment) if u.fragment else '')
            elif dest.is_file() and dest.is_relative_to(ROOT) and dest.stat().st_size<=4_000_000:
                # Supporting links are literal text snapshots, never executed.
                name=hashlib.sha256(str(dest.relative_to(ROOT)).encode()).hexdigest()[:12]+'-'+dest.name+'.txt'
                (OUT/'sources'/name).write_bytes(dest.read_bytes())
                manifest['snapshot:'+str(dest.relative_to(ROOT))]=hashlib.sha256(dest.read_bytes()).hexdigest()
                a['href']='sources/'+name
                a['title']='Source snapshot: '+str(dest.relative_to(ROOT))
            else:
                note=soup.new_tag('span',attrs={'title':'Reference remains in the source repository: '+a['href']})
                note.string=a.get_text()+' (see source repository)'
                a.replace_with(note)
        for table in soup.find_all('table'):
            wrap=soup.new_tag('div',attrs={'class':'table-wrap','tabindex':'0','role':'region','aria-label':'Scrollable reference table'})
            table.wrap(wrap)
        nav+=f'<a href="#{key}">{escape(title)}</a>'
        pages.append(f'<article class="page" id="{key}" data-title="{escape(title)}"><p class="source">Source: {source} · SHA-256 {manifest[source][:12]}</p><details class="toc"><summary>On this page</summary>{"".join(toc)}</details>{soup}</article>')
    home='''<article class="page" id="start" data-title="Start here">
<p class="eyebrow">Group 7 capstone · Ayush Kadali, Parth Tupe, Krishna Khandelwal / learning edition / November 2026</p>
<h1>Follow the computation.<br>Understand the system.</h1>
<p class="lead">A practical guide to <em>Hardware-Aware Runtime for Controlled CPU/Accelerator Placement Analysis</em> (working title: Transparent Execution-Provider Runtime and Placement-Robustness Testbed), an explainable model-to-accelerator exploration system. Start with three operations. Follow every placement decision, memory transfer, command and correctness check.</p>
<div class="actions"><a class="action primary" href="#tour">Start the five-minute demo</a><a class="action" href="#learn">Read the learning guide</a></div>
<div class="notice"><strong>Software college release (November 2026). Physical hardware deferred.</strong> Bounded ONNX import with a trained example, shape-aware analytical estimates on declared architecture profiles, exact engine cross-checks, a command-level device queue, and original-model correctness checks are implemented and tested in software. Physical BeagleY-AI work is deferred by the owner; see the hardware roadmap. PlacementLens names only the compiler/runtime executable.</div>
<div class="pipeline" aria-label="Execution pipeline"><span>Input</span><span>Graph</span><span>Placement</span><span>Commands</span><span>Buffers + queue</span><span>Execution</span><span>Reference check</span><span>Evidence</span></div>
<div class="route"><a href="#learn"><span>01 / Understand</span><strong>Learn the system</strong><span>24 modules, worked examples, exercises and separate answers, a seven-day defense route and a self-assessment checklist.</span></a><a href="#demo"><span>02 / Run</span><strong>Reproduce it</strong><span>Build commands, integrated execution, measured studies and the live workbench walkthrough.</span></a><a href="#acceptance"><span>03 / Defend</span><strong>Inspect the evidence</strong><span>Requirement-by-requirement status, actual measurements and clearly named remaining gates.</span></a></div>
<h2>How to use this book</h2><p>Choose a guide in the sidebar, or search for a concept such as “cancellation”, “residency” or “oracle”. Each chapter has a section index. Try the questions before opening the answer section. Use Print / PDF to save the chapter you are viewing.</p><p>The reading pages work offline. The live workbench needs the local server running. This book is generated from the repository’s Markdown; rebuild it after source documentation changes.</p>
</article>'''
    demo='''<article class="page" id="tour" data-title="Five-minute demo"><p class="eyebrow">A real captured CLI run / software backend</p><h1>One graph. Two placements.</h1><p class="lead">The numbers below come from fresh runs of the existing executable. This walkthrough replays their explanation; use the live workbench to run your own experiment.</p>
<div class="step-controls"><button id="previous">Previous step</button><span id="step-count" aria-live="polite"></span><button id="next">Next step</button></div>
<section class="stage"><h2>1. Describe the calculation</h2><p>Multiply A by B, replace negative values with zero, then add D. Placement changes where these operations run; the required answer stays the same.</p><pre>A = [[1, -2, 3], [0, 4, -1]]
B = [[2, 1], [1, -3], [-2, 2]]
D = [[1, 1], [1, 1]]

MatMul(A, B) → ReLU → Add(D) → Y</pre><p class="result">Expected Y = [[1, 14], [7, 1]]</p></section>
<section class="stage"><h2>2. The device cannot run ReLU</h2><p>The maximal-capability policy places MatMul and Add on the logical device. ReLU falls back to CPU. <strong>ACA</strong> means device → CPU → device.</p><div class="pipeline"><span>A · MatMul</span><span>C · ReLU</span><span>A · Add</span></div><p>That interruption introduces transfers between the two execution domains.</p><p class="result">6 copies · 112 bytes · 20 commands</p><p>80 bytes host→device, 32 bytes device→host; two device regions and two synchronizations. The independent reference check passed.</p></section>
<section class="stage"><h2>3. Enable device ReLU support</h2><p>With only capability changed, strict maximal placement becomes <strong>AAA</strong>. The intermediate tensors stay on the device between operations.</p><div class="pipeline"><span>A · MatMul</span><span>A · ReLU</span><span>A · Add</span></div><p class="result">4 copies · 80 bytes · 15 commands</p><p>64 bytes host→device, 16 bytes device→host; one device region and one synchronization. Output remains [[1, 14], [7, 1]]. Fewer transfers is an exact structural observation; it does not establish hardware speedup.</p></section>
<section class="stage"><h2>4. Exercise the integrated runtime</h2><p>Now combine buffer reuse, mirrored input A, tiled MatMul with a 48-byte scratch budget, and asynchronous host submission.</p><pre>build/placementlens verify --graph examples/fragmented.plg \\
  --relu-supported --reuse-buffers --resident-input A \\
  --tile-size 2 --local-bytes 48 --async</pre><p class="result">8 feasible placements verified · 0 numerical failures</p><p>All 128 emitted commands were exercised across these placements. The worker owns a whole graph submission; this does not demonstrate command overlap or accelerator hardware.</p></section>
<section class="stage"><h2>5. Try it in the workbench</h2><ol><li>Choose tensor width 2 and leave device ReLU support off.</li><li>Run experiment, then select <strong>Strict maximal supported</strong>.</li><li>Select the activation node. Read its rejection reason and linked commands.</li><li>Choose <strong>Use as baseline</strong>, enable <strong>Device supports ReLU</strong>, and run again.</li><li>Compare COPY commands, bytes, and regions. Inspect measured times separately.</li></ol><p>The workbench generates its own square-matrix inputs, so its byte totals differ from the rectangular CLI example above. Both illustrate the same boundary effect.</p><a class="action primary" href="http://127.0.0.1:8771/" target="_blank" rel="noopener">Open live workbench</a><p class="small">Local service required. Start it from the project directory with:</p><pre>python3 workbench/server.py --binary build/placementlens \\
  --out artifacts/onnx-cnn-2026-10-04/workbench-final --port 8771</pre><p><a href="#demo">Full reproduction guide</a> · <a href="#completion">What is still unfinished?</a></p></section>
<details><summary>Captured output and provenance</summary>CAPTURED</details></article>'''
    captured=[]
    evidence=ROOT/'artifacts/html-guide-demo-2026-10-04'
    for name in ('inspect','fragmented','supported','combined','manifest'):
        ext='.json' if name=='manifest' else '.txt'
        text=(evidence/(name+ext)).read_text()
        captured.append(f'<h3>{name}</h3><pre>{escape(text)}</pre>')
    demo=demo.replace('CAPTURED',''.join(captured))
    output=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Capstone learning guide</title><style>{CSS}</style></head><body><a class="skip" href="#content">Skip to reading</a><div class="layout"><aside><p class="eyebrow">Systems field notes</p><div class="brand"><a href="#start">Capstone<br>Learning Library</a></div><label for="search">Search all guides</label><input id="search" type="search" placeholder="Try: tensor, queue, oracle"><div id="results" aria-live="polite"></div><nav aria-label="Guide chapters">{nav}</nav><p class="small">Offline reading · source-linked<br>Software evidence ≠ board validation</p></aside><main id="content" tabindex="-1"><div class="top"><p>COMPILER / RUNTIME / EXPERIMENTS</p><button id="print">Print / PDF</button></div><noscript>JavaScript is off. All chapters are shown below; use your browser’s Find command.</noscript>{home}{demo}{''.join(pages)}<footer>Generated from current project documentation. Source files remain authoritative. No external scripts, fonts or tracking.</footer></main></div><script>{JS}</script></body></html>'''
    (OUT/'index.html').write_text(output)
    (OUT/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'{OUT / "index.html"}: {len(DOCS)} guides, {len(output):,} characters')
    write_reading_edition(output)


if __name__=='__main__':
    main()
