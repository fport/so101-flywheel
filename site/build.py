"""Renders docs/guide/{en,tr}/*.md to site/public/ (English at /, Turkish at /tr/).

    cd python && uv run python ../site/build.py
    cd site && npx wrangler deploy

The Markdown files are the only source: GitHub shows them as docs, this turns the same files into the site.
"""

import html
import re
import shutil
from pathlib import Path

import markdown
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GUIDE, PUBLIC, STATIC = ROOT / "docs" / "guide", ROOT / "site" / "public", ROOT / "site" / "static"
GITHUB = "https://github.com/fport/so101-flywheel"
PAGES = ["index", "cell", "setup", "inspect", "record", "data-quality", "review", "train", "rollout", "next", "concepts"]
UI = {
    "en": {"home": "The path", "prev": "Previous", "next": "Next", "menu": "Steps", "copy": "Copy", "copied": "Copied",
           "desc": "SO-101 zero to hero: build a cell, record demonstrations, train SmolVLA, run it on the arm."},
    "tr": {"home": "Yol haritası", "prev": "Önceki", "next": "Sonraki", "menu": "Adımlar", "copy": "Kopyala", "copied": "Kopyalandı",
           "desc": "SO-101 sıfırdan zirveye: hücreyi kur, gösterim kaydet, SmolVLA eğit, kolda çalıştır."},
}


def title_of(text):
    return re.search(r"^# (.+)$", text, re.M).group(1).strip()


def render(text):
    md = markdown.Markdown(extensions=["tables", "fenced_code", "attr_list", "toc"])
    body = md.convert(text)
    # links: other guide pages lose .md, repo files go to GitHub
    body = re.sub(r'href="([a-z-]+)\.md(#[^"]*)?"', lambda m: f'href="{"./" if m[1] == "index" else m[1]}{m[2] or ""}"', body)
    body = re.sub(r'href="(?:\.\./)+([^"]+)"', rf'href="{GITHUB}/blob/main/\1"', body)
    body = re.sub(r'<a href="http', '<a target="_blank" rel="noopener" href="http', body)
    body = body.replace("<li>[ ] ", '<li class="todo"><input type="checkbox"> ')
    return body.replace("<table>", '<div class="tbl"><table>').replace("</table>", "</table></div>")


def page(lang, name, text, titles):
    ui, i = UI[lang], PAGES.index(name)
    other = "tr" if lang == "en" else "en"
    here = "" if name == "index" else name
    url = lambda l, n: ("/" if l == "en" else "/tr/") + ("" if n == "index" else n)  # noqa: E731
    nav = "".join(
        f'<a href="{url(lang, n)}"{" class=on" if n == name else ""}>{html.escape(ui["home"] if n == "index" else titles[n])}</a>'
        for n in PAGES)
    pager = ""
    if i > 0:
        pager += f'<a class="prev" href="{url(lang, PAGES[i - 1])}"><i>← {ui["prev"]}</i>{html.escape(titles[PAGES[i - 1]])}</a>'
    if i < len(PAGES) - 1:
        pager += f'<a class="next" href="{url(lang, PAGES[i + 1])}"><i>{ui["next"]} →</i>{html.escape(titles[PAGES[i + 1]])}</a>'
    hero = '<img class="hero" src="/final.jpg" alt="The SO-101 quality control cell" width="1600" height="859">' if name == "index" else ""
    return f"""<!doctype html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(titles[name])} · SO-101 zero to hero</title>
<meta name="description" content="{html.escape(ui['desc'])}">
<link rel="alternate" hreflang="{other}" href="{url(other, name)}">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Ccircle cx='16' cy='16' r='15' fill='%2308080a' stroke='%23c8cbd1' stroke-width='1.5'/%3E%3Cpath d='M9 20l5-8 4 5 5-7' stroke='%23ff7a1a' stroke-width='2.2' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/style.css">
<script>if(new URLSearchParams(location.search).get("lang")==="{other}")location.replace("{url(other, name)}")</script>
</head>
<body data-copy="{ui['copy']}" data-copied="{ui['copied']}">
<header>
  <a class="mark" href="{url(lang, 'index')}"><b>SO-101</b> zero to hero</a>
  <span class="sp"></span>
  <a class="gh" href="{GITHUB}" target="_blank" rel="noopener">GitHub</a>
  <span class="lang"><a href="{url('en', name)}"{' class=on' if lang == 'en' else ''}>EN</a><a href="{url('tr', name)}"{' class=on' if lang == 'tr' else ''}>TR</a></span>
</header>
<div class="wrap">
<nav><details open><summary>{ui['menu']}</summary>{nav}</details></nav>
<main>
{hero}
{render(text)}
<div class="pager">{pager}</div>
</main>
</div>
<script src="/site.js"></script>
</body>
</html>
"""


def main():
    shutil.rmtree(PUBLIC, ignore_errors=True)
    shutil.copytree(STATIC, PUBLIC)
    Image.open(ROOT / "docs" / "media" / "final.png").convert("RGB").resize((1600, 859)).save(PUBLIC / "final.jpg", quality=86)
    for lang in ("en", "tr"):
        texts = {n: (GUIDE / lang / f"{n}.md").read_text() for n in PAGES}
        titles = {n: title_of(t) for n, t in texts.items()}
        out = PUBLIC if lang == "en" else PUBLIC / "tr"
        out.mkdir(exist_ok=True)
        for n, t in texts.items():
            (out / f"{n}.html").write_text(page(lang, n, t, titles))
    print(f"{len(PAGES)} pages × 2 languages → {PUBLIC}")


if __name__ == "__main__":
    main()
