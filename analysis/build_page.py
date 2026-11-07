"""Builds the paper's landing page (docs/index.html) from the paper sources, so the page and the paper
cannot disagree: the abstract is rendered from paper/sections/abstract.tex with the numbers from
paper/generated/values.tex, and the reference metadata from paper/references.bib.

Usage (from the repository root): python analysis/build_page.py
The page carries Highwire Press (citation_*) and Dublin Core tags for Google Scholar and a schema.org
ScholarlyArticle record.
"""
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = {
    "version": "2.0.0",
    "date": "2026-10-05",
    "pdf": "null-zoo-v2.0.0.pdf",
    "concept_doi": "10.5281/zenodo.23018987",
    "previous": [("Version 1.0.0 (28 September 2026)", "null-zoo-v1.0.0.pdf", "10.5281/zenodo.23018988")],
    "url": "https://arhancanli.github.io/null-zoo/",
    "repo": "https://github.com/arhancanli/null-zoo",
    "keywords": ["backtest overfitting", "data snooping", "multiple testing", "superior predictive ability",
                 "Reality Check", "Sharpe ratio", "deflated Sharpe ratio", "bootstrap", "simulation study",
                 "pre-registration", "quantitative finance"],
}

ACCENTS = {"'": "\u0301", "`": "\u0300", "^": "\u0302", '"': "\u0308", "~": "\u0303", "v": "\u030c", "c": "\u0327", "H": "\u030b"}


def latex_text(s):
    """Plain Unicode text from the small subset of LaTeX the paper's abstract and references use."""
    import unicodedata
    s = s.replace("\\ldots", "\u2026").replace("\\&", "&").replace("\\%", "%")
    s = re.sub(r"\\(['`^\"~vcH])\s*\{?\\?([A-Za-z])\}?", lambda m: unicodedata.normalize("NFC", m.group(2) + ACCENTS[m.group(1)]), s)
    s = re.sub(r"\\(texttt|emph|textbf|textit|mathrm|text)\{([^{}]*)\}", r"\2", s)
    s = s.replace("\\noindent", "").replace("---", "\u2014").replace("--", "\u2013")
    s = re.sub(r"\$([^$]*)\$", lambda m: m.group(1).replace("-", "\u2212").replace("\\phi", "\u03c6").replace("\\le", "\u2264").replace("\\ge", "\u2265"), s)
    s = s.replace("~", "\u00a0").replace("\\,", "\u2009").replace("\\ ", " ")
    s = re.sub(r"[{}]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def values():
    out = {}
    for line in (ROOT / "paper/generated/values.tex").read_text().splitlines():
        m = re.match(r"\\nzdef\{([^}]*)\}\{(.*)\}$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def abstract(vals):
    tex = (ROOT / "paper/sections/abstract.tex").read_text()
    tex = re.sub(r"\\(begin|end)\{abstract\}", "", tex)
    tex = re.sub(r"\\nz\{([^}]*)\}", lambda m: vals[m.group(1)], tex)
    return latex_text(tex)


def title():
    tex = (ROOT / "paper/main.tex").read_text()
    t = re.search(r"\\title\{(.*?)\}\n", tex).group(1).replace("\\\\", " ")
    return latex_text(t)


def bib_entries():
    bib = (ROOT / "paper/references.bib").read_text()
    entries = []
    for kind, key, body in re.findall(r"@(\w+)\{([^,]+),(.*?)\n\}", bib, re.S):
        f = {}
        for name in ("author", "title", "journal", "howpublished", "year", "doi"):
            m = re.search(name + r"\s*=\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", body)
            if m:
                f[name] = latex_text(m.group(1))
        entries.append(f)
    return entries


def main():
    vals = values()
    ttl, abs_text = title(), abstract(vals)
    esc = lambda s: html.escape(s, quote=True)  # noqa: E731
    metas = [
        ("citation_title", ttl), ("citation_author", "Canli, Arhan"), ("citation_author_institution", "Canli Capital"),
        ("citation_publication_date", PAGE["date"].replace("-", "/")), ("citation_online_date", PAGE["date"].replace("-", "/")),
        ("citation_pdf_url", PAGE["url"] + PAGE["pdf"]), ("citation_abstract_html_url", PAGE["url"]),
        ("citation_doi", PAGE["concept_doi"]), ("citation_technical_report_institution", "Canli Capital"),
        ("citation_technical_report_number", f"Null Zoo v{PAGE['version']}"), ("citation_language", "en"),
    ] + [("citation_keywords", k) for k in PAGE["keywords"]]
    for e in bib_entries():
        parts = [f"citation_title={e.get('title', '')}"]
        if e.get("author"):
            parts.append(f"citation_author={e['author'].split(' and ')[0]}")
        if e.get("year"):
            parts.append(f"citation_publication_date={e['year']}")
        venue = e.get("journal") or e.get("howpublished")
        if venue:
            parts.append(f"citation_journal_title={venue}")
        if e.get("doi"):
            parts.append(f"citation_doi={e['doi']}")
        metas.append(("citation_reference", "; ".join(parts)))
    dc = [("DC.title", ttl), ("DC.creator", "Canli, Arhan"), ("DC.date", PAGE["date"]), ("DC.identifier", "doi:" + PAGE["concept_doi"]),
          ("DC.type", "Text.Preprint"), ("DC.language", "en"), ("DC.rights", "CC BY 4.0")]
    ld = {"@context": "https://schema.org", "@type": "ScholarlyArticle", "headline": ttl, "name": ttl,
          "author": {"@type": "Person", "name": "Arhan Canli", "url": "https://github.com/arhancanli", "identifier": "https://orcid.org/0009-0004-4138-7907"},
          "datePublished": PAGE["date"], "version": PAGE["version"], "abstract": abs_text, "keywords": ", ".join(PAGE["keywords"]),
          "identifier": "https://doi.org/" + PAGE["concept_doi"], "url": PAGE["url"], "license": "https://creativecommons.org/licenses/by/4.0/",
          "isAccessibleForFree": True, "encoding": {"@type": "MediaObject", "contentUrl": PAGE["url"] + PAGE["pdf"], "encodingFormat": "application/pdf"}}
    bibtex = ("@misc{canli2026nullzoo,\n  author    = {Canli, Arhan},\n  title     = {" + ttl + "},\n  year      = {2026},\n"
              "  version   = {" + PAGE["version"] + "},\n  publisher = {Zenodo},\n  doi       = {" + PAGE["concept_doi"] + "},\n"
              "  url       = {https://doi.org/" + PAGE["concept_doi"] + "}\n}")
    prev = "".join(f'<li>{esc(label)}: <a href="{esc(pdf)}">PDF</a> · DOI <a href="https://doi.org/{esc(doi)}">{esc(doi)}</a></li>' for label, pdf, doi in PAGE["previous"])
    style = (ROOT / "docs/index.html").read_text()
    style = re.search(r"<style>.*?</style>", style, re.S).group(0)
    out = ["<!DOCTYPE html>", '<html lang="en">', "<head>", '<meta charset="utf-8">', '<meta name="viewport" content="width=device-width, initial-scale=1">',
           f"<title>{esc(ttl)}</title>"]
    out += [f'<meta name="{n}" content="{esc(v)}">' for n, v in metas + dc]
    out += [f'<meta name="description" content="{esc(abs_text[:300])}">', f'<meta property="og:title" content="{esc(ttl)}">',
            '<meta property="og:type" content="article">', f'<meta property="og:url" content="{PAGE["url"]}">',
            f'<meta property="og:description" content="{esc(abs_text[:300])}">', f'<link rel="canonical" href="{PAGE["url"]}">',
            f'<script type="application/ld+json">{json.dumps(ld, ensure_ascii=False)}</script>', style, "</head>", "<body>", "<main>",
            f"<h1>{esc(ttl)}</h1>",
            f'<div class="meta">Arhan Canli · Canli Capital · 5 October 2026 · Preprint, version {PAGE["version"]} · DOI <a href="https://doi.org/{PAGE["concept_doi"]}">{PAGE["concept_doi"]}</a> (all versions)</div>',
            f'<div class="links"><a href="{PAGE["pdf"]}">PDF</a><a href="https://doi.org/{PAGE["concept_doi"]}">Zenodo</a><a href="{PAGE["repo"]}">Code and data</a><a href="{PAGE["repo"]}/blob/main/analysis/v2/prereg-v2.json">Pre-registration</a></div>',
            "<h2>Abstract</h2>", f"<p>{esc(abs_text)}</p>",
            "<h2>Keywords</h2>", f"<p>{esc(', '.join(PAGE['keywords']))}</p>",
            "<h2>Cite</h2>", f"<pre>{esc(bibtex)}</pre>",
            "<h2>Earlier versions</h2>", f"<ul>{prev}</ul>",
            "<h2>Licence</h2>", '<p>Text: <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Code: MIT.</p>',
            "</main>", "</body>", "</html>", ""]
    (ROOT / "docs/index.html").write_text("\n".join(out))
    print(f"wrote docs/index.html ({len(metas)} citation tags); abstract {len(abs_text)} characters")


if __name__ == "__main__":
    main()
