# -*- coding: utf-8 -*-
"""Preregistration hash check (Appendix B of the paper).

Re-computes the SHA-256 values that were written to the timestamped queue log (data/merdiven/kuyruk6_durum.txt)
when each preregistration addition was written, and compares them with the files in this folder.

  Ek 7, 7a, 7b : text of the addition = from its '## Ek ...' heading up to (not including) the next '## ' heading
  Ek 7c        : same, but without the trailing separator lines ('---', blank lines); also checks the SUPERSEDED hash
                 (the same text with the header time written as ~16:40 instead of ~15:50)
  85M run      : the whole file KANIT_ON_KAYIT.md

Usage: python verify_hashes.py            (run from anywhere; files are looked up next to this script)
"""
import hashlib
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "..", "results", "queue_status_log.txt")
if not os.path.exists(LOG):  # yerel Devrim klasoru duzeni
    LOG = os.path.join(HERE, "..", "d1_frank", "merdiven", "kuyruk6_durum.txt")
ON = os.path.join(HERE, "ON_KAYIT_MERDIVEN_2026-10-08.md")
KANIT = os.path.join(HERE, "KANIT_ON_KAYIT.md")
if not os.path.exists(ON):  # yerel Devrim klasoru duzeni
    ON = os.path.join(HERE, "..", "ON_KAYIT_MERDIVEN_2026-10-08.md")
    KANIT = os.path.join(HERE, "..", "KANIT_ON_KAYIT.md")


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


log = open(LOG, encoding="utf-8").read()
logged = {}
for m in re.finditer(r"^\[[^\]]+\] (EK7[abc]?) sha256 ([0-9a-f]{64})", log, re.M):
    logged.setdefault(m.group(1), []).append(m.group(2))
m = re.search(r"KANIT_ON_KAYIT\.md sha256 ([0-9a-f]{64})", log)
logged["KANIT"] = [m.group(1)] if m else []

t = open(ON, encoding="utf-8").read()
heads = [(mm.start(), mm.group(0)) for mm in re.finditer(r"(?m)^## .*$", t)]


def bolum(anahtar):
    """Ek bolumu: basliktan sonraki '## ' basligina kadar."""
    for i, (pos, h) in enumerate(heads):
        if re.match(r"## Ek %s \(" % anahtar, h):
            son = heads[i + 1][0] if i + 1 < len(heads) else len(t)
            return t[pos:son]
    raise SystemExit("bolum yok: Ek " + anahtar)


def varyantlar(s):
    yield "ham", s
    yield "rstrip", s.rstrip()
    yield "rstrip+LF", s.rstrip() + "\n"
    # ayirac satirlarini ('---') da at
    s2 = re.sub(r"(\n\s*-{3,}\s*)+\s*$", "", s.rstrip())
    yield "ayiracsiz", s2
    yield "ayiracsiz+LF", s2 + "\n"
    # otomatik uretilen bolumun HTML isaretcisini ('<!-- K6-SONUC-BAS -->') da at
    s3 = re.sub(r"(\n\s*(<!--.*?-->|-{3,})\s*)+\s*$", "", s.rstrip(), flags=re.S)
    yield "ayirac+isaretci-siz", s3
    yield "ayirac+isaretci-siz+LF", s3 + "\n"


tamam = True
for ek, ad in (("7", "EK7"), ("7a", "EK7a"), ("7b", "EK7b"), ("7c", "EK7c")):
    b = bolum(ek)
    bulunan = None
    for ad_v, s in varyantlar(b):
        h = sha(s)
        if h in logged.get(ad, []):
            bulunan = (ad_v, h, len(s.encode("utf-8")))
            break
    print("Ek %-3s kayitli hash(ler): %s -> %s" % (ek, ",".join(x[:8] for x in logged.get(ad, [])),
          ("ESLESTI (%s, %d bayt, %s...)" % (bulunan[0], bulunan[2], bulunan[1][:8])) if bulunan else "ESLESMEDI"))
    tamam &= bool(bulunan)

# Ek 7c'nin gecersiz kilinan ilk hash'i: baslikta saat ~16:40 yazili hal
b = bolum("7c")
b_eski = b.replace("12 Ekim 2026 ~15:50", "12 Ekim 2026 ~16:40", 1)
eski_ok = False
for ad_v, s in varyantlar(b_eski):
    if sha(s) in logged.get("EK7c", []) and sha(s) != sha(next(varyantlar(b))[1]):
        eski_ok = True
        print("Ek 7c ilk (gecersiz) hash: ESLESTI ('~16:40' basligiyla, %s, %s...)" % (ad_v, sha(s)[:8]))
        break
if not eski_ok:
    print("Ek 7c ilk (gecersiz) hash: ESLESMEDI")
tamam &= eski_ok

k = hashlib.sha256(open(KANIT, "rb").read()).hexdigest()
ok = k in logged["KANIT"]
print("KANIT_ON_KAYIT.md (tum dosya): %s... -> %s" % (k[:8], "ESLESTI" if ok else "ESLESMEDI"))
tamam &= ok
print("SONUC:", "TUM HASH'LER ESLESTI" if tamam else "UYUSMAZLIK VAR")
sys.exit(0 if tamam else 1)
