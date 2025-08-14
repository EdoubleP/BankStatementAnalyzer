import argparse
import json
import os
import re
import sys
from datetime import datetime
from dateutil import parser as dtparse

import pdfplumber
import pandas as pd

# ------------------ regex ------------------
# Accept amounts like 12.34 or .09
MONEY_PAT = re.compile(
    r'[-+]?\$?\s?((?:\d{1,3}(?:,\d{3})*|\d+)?(?:\.\d{2}))'
)
MONEY_SIGNED = re.compile(
    r'\$?\s*((?:\d{1,3}(?:,\d{3})*|\d+)?(?:\.\d{2}))\s*(-)?'
)
DATE_PAT = re.compile(r'(\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b[A-Za-z]{3,9}\s+\d{1,2},\s*\d{4}\b)')
MD_SLASH = re.compile(r'^\s*(\d{2}/\d{2})\b')
MD_ANY   = re.compile(r'^\s*(\d{1,2}[/-]\d{1,2})\b')

# ------------------ config ------------------
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ------------------ helpers ------------------
def try_parse_date(s, default_year=None):
    try:
        d = dtparse.parse(str(s), fuzzy=True)
        if default_year and re.fullmatch(r'\d{1,2}[/-]\d{1,2}', str(s).strip()):
            d = d.replace(year=default_year)
        return d
    except Exception:
        return None

def normalize_amount(s):
    if s is None:
        return None
    txt = str(s).replace(',', '').replace('$', '').strip()
    if txt in ('', '-', '—'):
        return None
    try:
        return float(txt)
    except Exception:
        m = MONEY_PAT.search(str(s))
        if m:
            try:
                return float(m.group(1).replace(',', ''))
            except Exception:
                return None
        return None

def guess_company_name(full_text):
    lines = [l.strip() for l in (full_text or "").splitlines() if l.strip()]
    header = lines[:80]
    ent_tokens = (' llc', ' inc', ' corp', ' company', ' ltd', ' co.')
    for l in header:
        low = l.lower()
        if any(x in low for x in [
            'statement','account','page','member fdic',
            'chase','bank of america','keybank',
            'east cambridge savings','investors bank','citizens bank',
            'regions bank','regions.com','banner bank',"banner's best checking",
            'independent bank'
        ]):
            continue
        if any(t in low for t in ent_tokens) and re.search(r'[A-Za-z]', l):
            return l
    for l in header:
        low = l.lower()
        if any(x in low for x in ['statement','account','page','balance','period','deposits','withdrawals']):
            continue
        if re.search(r'[A-Za-z]', l):
            return l
    return ""

def extract_statement_period(text):
    # Matches e.g. "July 1, 2022 - July 31, 2022" OR "3/01/23 thru 4/02/23"
    sep = r'(?:-+|–|—|\bto\b|\bthru\b|\bthrough\b)'
    m = re.search(r'(' + DATE_PAT.pattern + r')\s*' + sep + r'\s*(' + DATE_PAT.pattern + r')', text, re.I)
    if m:
        d1 = try_parse_date(m.group(1))
        d2 = try_parse_date(m.group(2))
        if d1 and d2:
            return d1.date(), d2.date()
    m2 = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}', text, re.I)
    if m2:
        d = dtparse.parse(m2.group(0))
        start = d.replace(day=1).date()
        end = (pd.Timestamp(start) + pd.offsets.MonthEnd(0)).date()
        return start, end
    return None, None

def extract_plain_text(pdf):
    return "\n".join([(p.extract_text() or "") for p in pdf.pages])

def find_section(text, header, next_headers):
    low = text.lower()
    h = header.lower()
    start = low.find(h)
    if start == -1:
        return ""
    next_pos = len(text)
    for nh in next_headers:
        i = low.find(nh.lower(), start + len(h))
        if i != -1:
            next_pos = min(next_pos, i)
    return text[start:next_pos]

def parse_amount_from_block(block_text):
    amounts = MONEY_PAT.findall(block_text)
    if not amounts:
        return None
    amt = amounts[-1]
    try:
        return float(amt.replace(',', ''))
    except Exception:
        return None

# ================== BANK-SPECIFIC PARSERS ==================

# ---- Independent Bank (NEW) ----
INDEP_HEADERS = [
    "DEPOSITS AND OTHER CREDITS",
    "OTHER DEBITS",
    "CHECKS IN NUMBER ORDER",
    "DAILY BALANCE SUMMARY",
    "SERVICE CHARGE SUMMARY"
]
INDEP_CHECK_TWOCOL = re.compile(r'(\d{1,2}/\d{2})\s+(\d{3,}\*?)\s+\$?([0-9,]+\.\d{2})')

def parse_indep_block(block_text, year, sign, section_name):
    """Group by lines that start with mm/dd; amount = last money token; multi-line friendly; ignore 'Total' lines."""
    if not block_text:
        return []
    lines = [l.rstrip() for l in block_text.splitlines()]
    if lines:
        lines = lines[1:] if lines[0].strip() else lines
    txns, current = [], []

    def is_total(s): return s.strip().lower().startswith("total ")

    def flush(block):
        if not block:
            return None
        first = block[0]
        if is_total(first):
            return None
        m = MD_SLASH.match(first)
        if not m:
            return None
        md = m.group(1)
        d = try_parse_date(md, default_year=year)
        if not d:
            return None
        text = " ".join([b.strip() for b in block if b.strip() and not is_total(b)])
        amt = parse_amount_from_block(text)
        if amt is None:
            return None
        desc = text[len(md):].strip(" -:•")
        return {'date': d.date(), 'description': desc, 'amount': amt*sign, 'balance': None, 'section': section_name}

    for line in lines:
        if not line.strip():
            continue
        if is_total(line):
            if current:
                rec = flush(current)
                if rec: txns.append(rec)
                current = []
            continue
        if MD_SLASH.match(line):
            if current:
                rec = flush(current)
                if rec: txns.append(rec)
                current = []
            current.append(line)
        else:
            if current:
                current.append(line)
    if current:
        rec = flush(current)
        if rec: txns.append(rec)
    return txns

def parse_indep_checks(checks_text, year):
    """Each line can contain two triplets; parse all."""
    if not checks_text:
        return []
    lines = [l.strip() for l in checks_text.splitlines()]
    out = []
    for line in lines:
        low = line.lower()
        if low.startswith("date") or low.startswith("checks in") or low.startswith("listed in") or low.startswith("(*)") or low.startswith("total"):
            continue
        for md, chk, amt in INDEP_CHECK_TWOCOL.findall(line):
            d = try_parse_date(md, default_year=year)
            if not d:
                continue
            try:
                amount = float(amt.replace(',', ''))
            except Exception:
                continue
            out.append({'date': d.date(), 'description': f"Check {chk}".replace('*',''), 'amount': -amount, 'balance': None, 'section': 'CHECKS'})
    return out

def parse_indep_daily_balances(section_text, year):
    daily = {}
    if not section_text:
        return daily
    for md, bal in re.findall(r'(\d{2}/\d{2})\s+\$?([0-9,]+\.\d{2})', section_text):
        d = try_parse_date(md, default_year=year)
        if not d:
            continue
        try:
            daily[d.date()] = float(bal.replace(',', ''))
        except Exception:
            pass
    return daily

def parse_independent(full_text, period_year):
    def sec(h):
        nxt = [x for x in INDEP_HEADERS if x != h]
        return find_section(full_text, h, nxt)
    deposits = sec("DEPOSITS AND OTHER CREDITS")
    debits   = sec("OTHER DEBITS")
    checks   = sec("CHECKS IN NUMBER ORDER")
    daily    = sec("DAILY BALANCE SUMMARY")
    # service_chg = sec("SERVICE CHARGE SUMMARY")  # usually duplicated as an "OTHER DEBITS" line

    txns = []
    txns += parse_indep_block(deposits, period_year, +1, "DEPOSITS AND OTHER CREDITS")
    txns += parse_indep_block(debits,   period_year, -1, "OTHER DEBITS")
    txns += parse_indep_checks(checks, period_year)
    daily_bal = parse_indep_daily_balances(daily, period_year)
    return txns, daily_bal

# ---- Regions ----
REGIONS_HEADERS = ["DEPOSITS & CREDITS","WITHDRAWALS","FEES","CHECKS","DAILY BALANCE SUMMARY"]

def parse_regions_block(block_text, year, sign, section_name):
    if not block_text:
        return []
    lines = [l.rstrip() for l in block_text.splitlines()]
    if lines:
        lines = lines[1:] if lines[0].strip() else lines
    txns, current = [], []
    def is_total_line(s): return s.strip().lower().startswith("total ")
    def flush(block):
        if not block: return None
        first = block[0]
        if is_total_line(first): return None
        m = MD_SLASH.match(first)
        if not m: return None
        md = m.group(1); d = try_parse_date(md, default_year=year)
        if not d: return None
        text = " ".join([b.strip() for b in block if b.strip() and not is_total_line(b)])
        amt = parse_amount_from_block(text)
        if amt is None: return None
        desc = text[len(md):].strip(" -:•")
        return {'date': d.date(), 'description': desc, 'amount': amt*sign, 'balance': None, 'section': section_name}
    for line in lines:
        if not line.strip(): continue
        if is_total_line(line):
            if current:
                rec = flush(current); 
                if rec: txns.append(rec)
                current = []
            continue
        if MD_SLASH.match(line):
            if current:
                rec = flush(current); 
                if rec: txns.append(rec)
                current = []
            current.append(line)
        else:
            if current: current.append(line)
    if current:
        rec = flush(current); 
        if rec: txns.append(rec)
    return txns

REGIONS_CHECK = re.compile(r'^\s*(\d{2}/\d{2})\s+(\d{3,})\s+\*?\s+\$?([0-9,]+\.\d{2})')

def parse_regions_checks(checks_text, year):
    if not checks_text: return []
    lines = [l.strip() for l in checks_text.splitlines()]
    out = []
    for line in lines:
        low = line.lower()
        if low.startswith("date") or low.startswith("total "): continue
        m = REGIONS_CHECK.match(line)
        if not m: continue
        md, chk, amt = m.groups()
        d = try_parse_date(md, default_year=year)
        if not d: continue
        try: amount = float(amt.replace(',', ''))
        except Exception: continue
        out.append({'date': d.date(), 'description': f"Check {chk}", 'amount': -amount, 'balance': None, 'section': 'CHECKS'})
    return out

def parse_regions_daily_balances(section_text, year):
    daily = {}
    if not section_text: return daily
    for line in section_text.splitlines():
        for md, bal in re.findall(r'(\d{2}/\d{2})\s+\$?([0-9,]+\.\d{2})', line):
            d = try_parse_date(md, default_year=year)
            if not d: continue
            try: daily[d.date()] = float(bal.replace(',', ''))
            except Exception: pass
    return daily

def parse_regions(full_text, period_year):
    def sec(h):
        nxt = [x for x in REGIONS_HEADERS if x != h]
        return find_section(full_text, h, nxt)
    deposits = sec("DEPOSITS & CREDITS")
    withdrawals = sec("WITHDRAWALS")
    fees = sec("FEES")
    checks = sec("CHECKS")
    daily = sec("DAILY BALANCE SUMMARY")
    txns = []
    txns += parse_regions_block(deposits, period_year, +1, "DEPOSITS & CREDITS")
    txns += parse_regions_block(withdrawals, period_year, -1, "WITHDRAWALS")
    txns += parse_regions_block(fees, period_year, -1, "FEES")
    txns += parse_regions_checks(checks, period_year)
    daily_bal = parse_regions_daily_balances(daily, period_year)
    return txns, daily_bal

# ---- Investors/Citizens ----
def parse_investors(full_text, period_year):
    lines = full_text.splitlines()
    in_detail, blocks, current = False, [], []
    for raw in lines:
        line = raw.rstrip(); low = line.lower()
        if 'account activity detail' in low: in_detail = True; continue
        if in_detail and ('check register' in low or low.startswith('check register')): in_detail = False
        if not in_detail: continue
        if MD_SLASH.match(line):
            if current: blocks.append("\n".join(current)); current = []
            current.append(line)
        else:
            if current: current.append(line)
    if current: blocks.append("\n".join(current))
    txns = []
    def parse_block(block_text):
        m = MD_SLASH.match(block_text.splitlines()[0])
        if not m: return None
        md = m.group(1); d = try_parse_date(md, default_year=period_year)
        if not d: return None
        tokens = []
        for mm in MONEY_SIGNED.finditer(block_text):
            val = float(mm.group(1).replace(',', '')); neg = True if mm.group(2) else False
            tokens.append((val, neg, mm.start()))
        if len(tokens) < 2: return None
        bal = tokens[-1][0]
        prev = tokens[:-1]
        neg_prev = [t for t in prev if t[1]]
        if neg_prev: amt = -neg_prev[-1][0]
        else:
            cand = prev[-1]; amt = -cand[0] if cand[1] else cand[0]
        desc = block_text[len(md):].strip(" -:\n\t")
        desc = re.sub(r'\$?\s*(?:\d{1,3}(?:,\d{3})*|\d+)?(?:\.\d{2})(?:-)?', '', desc).strip()
        if 'beginning balance' in desc.lower(): return None
        return {'date': d.date(), 'description': re.sub(r'\s+', ' ', desc),
                'amount': amt, 'balance': bal, 'section': 'ACCOUNT ACTIVITY DETAIL'}
    for b in blocks:
        rec = parse_block(b)
        if rec: txns.append(rec)
    daily = {}
    if txns:
        df = pd.DataFrame(txns)
        if 'balance' in df and df['balance'].notna().any():
            s = df[['date', 'balance']].dropna().groupby('date')['balance'].last()
            daily = {k: float(v) for k, v in s.to_dict().items()}
    return txns, daily

# ---- ECSB ----
def detect_header_row(df, must_have):
    for i in range(min(10, len(df))):
        row_text = " ".join([str(x or "") for x in df.iloc[i].tolist()]).lower()
        if all(tok in row_text for tok in must_have):
            return i
    return None

def parse_ecsb_tables(tables, year):
    txns = []
    for df in tables:
        hdr_idx = detect_header_row(df, must_have=['date','transaction','withdrawal','deposit','balance'])
        if hdr_idx is None: continue
        data = df.iloc[hdr_idx+1:].copy()
        headers = [str(c).strip().lower() for c in df.iloc[hdr_idx].tolist()]
        data.columns = headers
        cols = list(data.columns)
        def col_like(cands):
            for c in cols:
                if any(x in c for x in cands): return c
            return None
        c_date = col_like(['date']); c_desc = col_like(['transaction'])
        c_with = col_like(['withdrawal']); c_dep = col_like(['deposit']); c_bal = col_like(['balance'])
        current = None
        def start_new(date_val, desc):
            return {'date': date_val, 'description': (desc or '').strip(),
                    'amount': None, 'balance': None, 'section': 'Ledger'}
        for _, row in data.iterrows():
            raw_date = str(row.get(c_date, '')).strip() if c_date else ''
            date_val = try_parse_date(raw_date, default_year=year) if raw_date else None
            desc_val = str(row.get(c_desc, '')).strip() if c_desc else ''
            w = normalize_amount(row.get(c_with, '')) if c_with else None
            d = normalize_amount(row.get(c_dep, '')) if c_dep else None
            b = normalize_amount(row.get(c_bal, '')) if c_bal else None
            continuation = (not raw_date) and (w is None) and (d is None) and (desc_val != '')
            if date_val is not None and raw_date:
                if current is not None and current.get('date'): txns.append(current)
                current = start_new(date_val.date(), desc_val)
            elif continuation:
                if current: current['description'] = (current['description'] + " " + desc_val).strip()
                continue
            elif current is None:
                continue
            if d is not None or w is not None:
                current['amount'] = (d or 0.0) - (w or 0.0)
            if b is not None:
                current['balance'] = b
        if current is not None and current.get('date'): txns.append(current)
    out = []
    for t in txns:
        if t.get('date') and (t.get('description') or t.get('amount') is not None):
            out.append(t)
    return out, {}

# ---- KeyBank ----
KEYBANK_HEADERS = ["Additions", "Subtractions", "Fees and charges", "Fees and\ncharges"]

def parse_keybank_block_transactions(block_text, year, sign, section_name):
    if not block_text: return []
    lines = [l.rstrip() for l in block_text.splitlines()]
    if lines: lines = lines[1:] if lines[0].strip() else lines
    txns, current = [], []
    def flush(block):
        if not block: return None
        m = MD_ANY.match(block[0]); 
        if not m: return None
        md = m.group(1)
        d = try_parse_date(md, default_year=year)
        if not d: return None
        text = " ".join([b.strip() for b in block if b.strip()])
        amount = parse_amount_from_block(text)
        if amount is None: return None
        desc = text[len(md):].strip(" -:•")
        return {'date': d.date(), 'description': desc, 'amount': amount*sign, 'balance': None, 'section': section_name}
    for line in lines:
        if not line.strip(): continue
        if MD_ANY.match(line):
            if current:
                rec = flush(current); 
                if rec: txns.append(rec)
                current = []
            current.append(line)
        else:
            if current: current.append(line)
    if current:
        rec = flush(current); 
        if rec: txns.append(rec)
    return txns

CHECK_ANY = re.compile(r'\*?\s?(\d{3,})\s+(\d{1,2}[/-]\d{1,2})\s+\$?([0-9,]+\.\d{2})')

def parse_keybank_checks(subtractions_text, year):
    if not subtractions_text: return []
    low = subtractions_text.lower()
    i_checks = low.find("paper checks"); i_withdr = low.find("withdrawals")
    if i_checks == -1: return []
    paper_block = subtractions_text[i_checks:(i_withdr if i_withdr != -1 else len(subtractions_text))]
    out = []
    for chk, md, amt in CHECK_ANY.findall(paper_block):
        d = try_parse_date(md, default_year=year); 
        if not d: continue
        try: amount = float(amt.replace(',', ''))
        except Exception: continue
        out.append({'date': d.date(), 'description': f"Check {chk}", 'amount': -amount, 'balance': None, 'section': 'Paper Checks'})
    return out

def parse_keybank_withdrawals(subtractions_text, year):
    if not subtractions_text: return []
    low = subtractions_text.lower()
    i_withdr = low.find("withdrawals")
    if i_withdr == -1: return []
    w_block = subtractions_text[i_withdr:len(subtractions_text)]
    return parse_keybank_block_transactions(w_block, year, sign=-1, section_name="Withdrawals")

def parse_keybank_additions(additions_text, year):
    if not additions_text: return []
    return parse_keybank_block_transactions(additions_text, year, sign=+1, section_name="Deposits")

def parse_keybank_fees(text, year):
    if not text: return []
    lines = [l.rstrip() for l in text.splitlines()]
    if lines: lines = lines[1:] if lines[0].strip() else lines
    txns, current = [], []
    def flush(block):
        if not block: return None
        m = MD_ANY.match(block[0]); 
        if not m: return None
        md = m.group(1); d = try_parse_date(md, default_year=year)
        if not d: return None
        text = " ".join([b.strip() for b in block if b.strip()])
        amount = parse_amount_from_block(text)
        if amount is None: return None
        if amount > 0: amount = -amount
        desc = text[len(md):].strip(" -:•")
        return {'date': d.date(), 'description': desc, 'amount': amount, 'balance': None, 'section': 'Fees and charges'}
    for line in lines:
        if not line.strip(): continue
        if MD_ANY.match(line):
            if current:
                rec = flush(current); 
                if rec: txns.append(rec)
                current = []
            current.append(line)
        else:
            if current: current.append(line)
    if current:
        rec = flush(current); 
        if rec: txns.append(rec)
    return txns

def parse_keybank(full_text, period_year):
    def section(h):
        heads = ["Additions", "Subtractions", "Fees and charges", "Fees and\ncharges"]
        nxt = [x for x in heads if x != h]
        return find_section(full_text, h, nxt)
    additions    = section("Additions")
    subtractions = section("Subtractions")
    fees         = section("Fees and charges") or section("Fees and\ncharges")
    txns  = []
    txns += parse_keybank_additions(additions, period_year)
    txns += parse_keybank_checks(subtractions, period_year)
    txns += parse_keybank_withdrawals(subtractions, period_year)
    txns += parse_keybank_fees(fees, period_year)
    return txns, {}

# ---- BofA ----
BOFA_SECTION_HEADERS = [
    "Deposits and other credits",
    "Withdrawals and other debits",
    "Checks",
    "Service fees",
    "Daily ledger balances"
]

def parse_bofa_section_transactions(section_text, year, sign, section_name):
    if not section_text: return []
    lines = [l.rstrip() for l in section_text.splitlines()]
    if lines: lines = lines[1:] if lines[0].strip() else lines
    txns, current = [], []
    def flush(block):
        if not block: return None
        m = MD_SLASH.match(block[0]); 
        if not m: return None
        md = m.group(1); d = try_parse_date(md, default_year=year)
        if not d: return None
        block_text = " ".join([b.strip() for b in block if b.strip()])
        amount = parse_amount_from_block(block_text)
        if amount is None: return None
        desc = block_text[len(md):].strip(" -:•")
        return {'date': d.date(), 'description': desc, 'amount': amount*sign, 'balance': None, 'section': section_name}
    for line in lines:
        if not line.strip(): continue
        if MD_SLASH.match(line):
            if current:
                rec = flush(current); 
                if rec: txns.append(rec)
                current = []
            current.append(line)
        else:
            if current: current.append(line)
    if current:
        rec = flush(current); 
        if rec: txns.append(rec)
    return txns

CHECK_LINE = re.compile(r'^\s*(\d{3,})\D+(\d{2}/\d{2})\D+\$?([0-9,]+\.\d{2})')

def parse_bofa_checks(section_text, year):
    if not section_text: return []
    lines = [l.strip() for l in section_text.splitlines()]
    if lines: lines = lines[1:] if lines[0].strip() else lines
    out = []
    for line in lines:
        m = CHECK_LINE.match(line)
        if not m: continue
        chk, md, amt = m.groups()
        d = try_parse_date(md, default_year=year)
        if not d: continue
        try: amount = float(amt.replace(',', ''))
        except Exception: continue
        out.append({'date': d.date(), 'description': f"Check {chk}", 'amount': -amount, 'balance': None, 'section': 'Checks'})
    return out

def parse_bofa_daily_balances(section_text, year):
    daily = {}
    if not section_text: return daily
    for line in section_text.splitlines():
        for md, amt in re.findall(r'(\d{2}/\d{2})\s+\$?([0-9,]+\.\d{2})', line):
            d = try_parse_date(md, default_year=year)
            if d:
                try: daily[d.date()] = float(amt.replace(',', ''))
                except Exception: pass
    return daily

def parse_bofa(full_text, period_year):
    def sec(h):
        nxt = [x for x in BOFA_SECTION_HEADERS if x != h]
        return find_section(full_text, h, nxt)
    deposits   = sec("Deposits and other credits")
    withdrawals= sec("Withdrawals and other debits")
    checks     = sec("Checks")
    fees       = sec("Service fees")
    daily      = sec("Daily ledger balances")
    txns  = []
    txns += parse_bofa_section_transactions(deposits,    period_year, +1, "Deposits and other credits")
    txns += parse_bofa_section_transactions(withdrawals, period_year, -1, "Withdrawals and other debits")
    txns += parse_bofa_checks(checks, period_year)
    txns += parse_bofa_section_transactions(fees,        period_year, -1, "Service fees")
    daily_bal = parse_bofa_daily_balances(daily, period_year)
    return txns, daily_bal

# ---- Chase ----
CHASE_SECTION_HEADERS = [
    "DEPOSITS AND ADDITIONS",
    "ATM & DEBIT CARD WITHDRAWALS",
    "ATM & DEBIT CARD SUMMARY",
    "ELECTRONIC WITHDRAWALS",
    "OTHER WITHDRAWALS",
    "CHECKS PAID",
    "FEES",
    "DAILY ENDING BALANCE"
]

def parse_chase_section_transactions(section_text, year, sign, section_name):
    if not section_text:
        return []
    lines = [l.rstrip() for l in section_text.splitlines()]
    if lines:
        lines = lines[1:] if lines[0].strip() else lines
    txns, current = [], []
    def flush(block):
        if not block: return None
        m = MD_SLASH.match(block[0]); 
        if not m: return None
        md = m.group(1); d = try_parse_date(md, default_year=year)
        if not d: return None
        block_text = " ".join([b.strip() for b in block if b.strip()])
        amount = parse_amount_from_block(block_text)
        if amount is None: return None
        desc = block_text[len(md):].strip(" -:•")
        return {'date': d.date(), 'description': desc, 'amount': amount*sign, 'balance': None, 'section': section_name}
    for line in lines:
        if not line.strip(): continue
        if MD_SLASH.match(line):
            if current:
                rec = flush(current); 
                if rec: txns.append(rec)
                current = []
            current.append(line)
        else:
            if current: current.append(line)
    if current:
        rec = flush(current); 
        if rec: txns.append(rec)
    return txns

def parse_chase_checks_paid(section_text, year):
    if not section_text:
        return []
    lines = [l.strip() for l in section_text.splitlines()]
    if lines:
        lines = lines[1:] if lines[0].strip() else lines
    out = []
    for line in lines:
        m = CHECK_LINE.match(line)
        if not m: continue
        chk, md, amt = m.groups()
        d = try_parse_date(md, default_year=year)
        if not d: continue
        try: amount = float(amt.replace(',', ''))
        except Exception: continue
        out.append({'date': d.date(), 'description': f"Check {chk}", 'amount': -amount, 'balance': None, 'section': 'CHECKS PAID'})
    return out

def parse_chase_daily_balances(section_text, year):
    daily = {}
    if not section_text: return daily
    for line in section_text.splitlines():
        for md, amt in re.findall(r'(\d{2}/\d{2})\s+\$?([0-9,]+\.\d{2})', line):
            d = try_parse_date(md, default_year=year)
            if d:
                try: daily[d.date()] = float(amt.replace(',', ''))
                except Exception: pass
    return daily

def parse_chase(text, period_year):
    def section(h): nxt = [x for x in CHASE_SECTION_HEADERS if x != h]; return find_section(text, h, nxt)
    deposits   = section("DEPOSITS AND ADDITIONS")
    atm_debit  = section("ATM & DEBIT CARD WITHDRAWALS")
    electronic = section("ELECTRONIC WITHDRAWALS")
    other_wdr  = section("OTHER WITHDRAWALS")
    checks     = section("CHECKS PAID")
    fees       = section("FEES")
    daily      = section("DAILY ENDING BALANCE")
    txns  = []
    txns += parse_chase_section_transactions(deposits,   period_year, +1, "DEPOSITS AND ADDITIONS")
    txns += parse_chase_section_transactions(atm_debit,  period_year, -1, "ATM & DEBIT CARD WITHDRAWALS")
    txns += parse_chase_section_transactions(electronic, period_year, -1, "ELECTRONIC WITHDRAWALS")
    txns += parse_chase_section_transactions(other_wdr,  period_year, -1, "OTHER WITHDRAWALS")
    txns += parse_chase_checks_paid(checks, period_year)
    txns += parse_chase_section_transactions(fees,       period_year, -1, "FEES")
    daily_bal = parse_chase_daily_balances(daily, period_year)
    return txns, daily_bal

# ------------------ recurring debit detection & summary ------------------
def detect_recurring_debit(df):
    def norm_desc(s):
        s = re.sub(r'\s+', ' ', s or '').strip().lower()
        s = re.sub(r'\d+', '', s)
        return s

    debits = df[df['amount'] < 0].copy()
    if debits.empty:
        return None, None

    debits['norm'] = debits['description'].apply(norm_desc)
    candidates = []
    for key, grp in debits.groupby('norm'):
        if len(grp) >= 3:
            dates = sorted(list(grp['date']))
            deltas = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
            if not deltas:
                continue
            med = pd.Series(deltas).median()
            amt_mode = grp['amount'].mode()
            amt_mode_val = float(amt_mode.iloc[0]) if not amt_mode.empty else float(grp['amount'].iloc[-1])
            freq = ''
            if 1 <= med <= 2:   freq = 'daily'
            elif 6 <= med <= 8: freq = 'weekly'
            elif 13 <= med <= 16: freq = 'biweekly'
            elif 28 <= med <= 31: freq = 'monthly'
            if freq:
                candidates.append((freq, abs(amt_mode_val), key))
    if not candidates:
        return None, None
    order = {'daily':4, 'weekly':3, 'biweekly':2, 'monthly':1}
    candidates.sort(key=lambda x: (order.get(x[0], 0), x[1]), reverse=True)
    freq, pay_amt, key = candidates[0]
    return pay_amt, freq

def summarize(txns, cfg, daily_balances=None):
    df = pd.DataFrame(txns)
    if df.empty and not daily_balances:
        return {
            'Total Deposits': 0.0,
            'Number of Deposits': 0,
            'Number of Negative Days': 0,
            'Avg Daily Balance': None,
            'Number of Cash Advances': 0,
            'MCA Company Names': '',
            'Payment Amount': None,
            'Payment Frequency': '',
            'Amount Changed Over 3 Months': None,
            'MCA Fundings (Count)': 0,
            'MCA Fundings (Total)': 0.0,
            'MCA Funding (Largest)': None,
            'Large Deposits (Count)': 0,
            'Large Deposits (Total)': 0.0,
            'Largest Deposit (Any)': None
        }

    if not df.empty:
        deposits = df[df['amount'] > 0] if cfg.get('assume_deposit_if_positive', True) else df[df['amount'] >= 0.01]
        total_deposits = float(deposits['amount'].sum()) if not deposits.empty else 0.0
        num_deposits = int(len(deposits))
    else:
        total_deposits = 0.0
        num_deposits = 0

    neg_days = 0
    avg_bal = None
    if daily_balances:
        s = pd.Series(daily_balances)
        neg_days = int((s < 0).sum())
        avg_bal = float(s.mean()) if not s.empty else None
    elif (not df.empty) and 'balance' in df and df['balance'].notna().any():
        by_day = df[['date', 'balance']].dropna().groupby('date')['balance'].last()
        if not by_day.empty:
            neg_days = int((by_day < 0).sum())
            avg_bal = float(by_day.mean())

    mca_keys    = [k.lower() for k in cfg.get('mca_keywords', [])]
    mca_lenders = [k.lower() for k in cfg.get('mca_lenders', [])]
    fund_pos_terms = [k.lower() for k in cfg.get('mca_funding_positive_terms', [])]
    large_thr   = float(cfg.get('large_deposit_threshold', 10000.0))

    def is_mca_text(desc):
        l = (desc or '').lower()
        return any(k in l for k in (mca_keys + mca_lenders))

    def is_mca_funding_deposit(row):
        if row['amount'] is None or row['amount'] <= 0:
            return False
        text = (row['description'] or '').lower()
        return any(t in text for t in fund_pos_terms) or any(l in text for l in mca_lenders)

    if not df.empty:
        mca_rows = df[df['description'].fillna('').apply(is_mca_text)]
        num_cash_advances = int(len(mca_rows))

        mca_names_found = set()
        for desc in mca_rows['description'].dropna():
            low = desc.lower()
            for lender in mca_lenders:
                if lender in low:
                    mca_names_found.add(lender)
        mca_names_str = ", ".join(sorted(mca_names_found))

        pay_amt, pay_freq = detect_recurring_debit(df)

        mca_fund_deposits = df[df.apply(is_mca_funding_deposit, axis=1)]
        mca_fund_count = int(len(mca_fund_deposits))
        mca_fund_total = float(mca_fund_deposits['amount'].sum()) if mca_fund_count else 0.0
        mca_fund_largest = float(mca_fund_deposits['amount'].max()) if mca_fund_count else None

        large_deposits = df[(df['amount'] > 0) & (df['amount'] >= large_thr)]
        large_count = int(len(large_deposits))
        large_total = float(large_deposits['amount'].sum()) if large_count else 0.0
        largest_any_deposit = float(df[df['amount'] > 0]['amount'].max()) if (df['amount'] > 0).any() else None
    else:
        num_cash_advances = 0
        mca_names_str = ""
        pay_amt = None
        pay_freq = ""
        mca_fund_count = 0
        mca_fund_total = 0.0
        mca_fund_largest = None
        large_count = 0
        large_total = 0.0
        largest_any_deposit = None

    amt_change_3mo = None
    if (not df.empty) and (pay_amt is not None and pay_freq):
        debits = df[df['amount'] < 0].copy()
        debits['abs_amt'] = debits['amount'].abs()
        fam = debits[(debits['abs_amt'] >= 0.995*pay_amt) & (debits['abs_amt'] <= 1.005*pay_amt)]
        if not fam.empty:
            fam['month'] = fam['date'].apply(lambda d: f"{d.year}-{d.month:02d}")
            monthly = fam.groupby('month')['amount'].sum().sort_index()
            if len(monthly) >= 3:
                first3 = monthly.iloc[:3]
                amt_change_3mo = float(first3.iloc[-1] - first3.iloc[0])

    return {
        'Total Deposits': total_deposits,
        'Number of Deposits': num_deposits,
        'Number of Negative Days': neg_days,
        'Avg Daily Balance': avg_bal,
        'Number of Cash Advances': num_cash_advances,
        'MCA Company Names': mca_names_str,
        'Payment Amount': pay_amt,
        'Payment Frequency': pay_freq,
        'Amount Changed Over 3 Months': amt_change_3mo,
        'MCA Fundings (Count)': mca_fund_count,
        'MCA Fundings (Total)': mca_fund_total,
        'MCA Funding (Largest)': mca_fund_largest,
        'Large Deposits (Count)': large_count,
        'Large Deposits (Total)': large_total,
        'Largest Deposit (Any)': largest_any_deposit
    }

# ------------------ generic fallbacks ------------------
def extract_tables(pdf):
    all_tables = []
    for p in pdf.pages:
        try:
            tables = p.extract_tables()
            for t in tables:
                if not t: continue
                maxw = max(len(r) for r in t if r)
                clean = [(r + ['']*(maxw - len(r))) for r in t if r and len(r) >= maxw//2]
                if clean: all_tables.append(pd.DataFrame(clean))
        except Exception:
            continue
    return all_tables

def detect_bank_header(df):
    header_row_idx = None
    for i in range(min(8, len(df))):
        row_text = " ".join([str(x or "") for x in df.iloc[i].tolist()]).lower()
        if any(h in row_text for h in ['date','description','amount','debit','credit','balance']):
            header_row_idx = i; break
    return header_row_idx

def coerce_transactions_from_tables(tables):
    txns = []
    for df in tables:
        header_row_idx = detect_bank_header(df)
        if header_row_idx is None:
            continue
        data = df.iloc[header_row_idx+1:].copy()
        data.columns = [str(c).strip().lower() for c in df.iloc[header_row_idx].tolist()]
        cols = list(data.columns)
        def findcol(cands):
            for c in cols:
                if any(x in c for x in cands): return c
            return None
        c_date = findcol(['date'])
        c_desc = findcol(['description','memo','details','transaction'])
        c_amount = findcol(['amount','transaction amount','debit/credit'])
        c_balance = findcol(['balance','running'])
        c_debit = findcol(['withdrawal','withdrawals','debit'])
        c_credit = findcol(['deposit','deposits','credit','credits'])
        for _, row in data.iterrows():
            date = try_parse_date(row.get(c_date, '')) if c_date else None
            desc = str(row.get(c_desc, '')).strip() if c_desc else ""
            amt = None
            if c_amount:
                amt = normalize_amount(row.get(c_amount, ''))
            elif c_debit or c_credit:
                deb = normalize_amount(row.get(c_debit, '')) if c_debit else None
                cre = normalize_amount(row.get(c_credit, '')) if c_credit else None
                if deb is not None or cre is not None:
                    amt = (cre or 0.0) - (deb or 0.0)
            bal = normalize_amount(row.get(c_balance, '')) if c_balance else None
            if date and (desc or amt is not None):
                txns.append({'date': date.date(), 'description': desc, 'amount': amt, 'balance': bal})
    return txns

def coerce_transactions_from_text(text):
    txns = []
    for line in text.splitlines():
        if DATE_PAT.search(line) and MONEY_PAT.search(line):
            d = try_parse_date(line)
            amt = normalize_amount(line)
            if d and amt is not None:
                desc = re.sub(MONEY_PAT, '', line).strip()
                txns.append({'date': d.date(), 'description': desc, 'amount': amt, 'balance': None})
    return txns

# ------------------ per-file pipeline ------------------
def process_pdf(path, cfg):
    with pdfplumber.open(path) as pdf:
        full_text = extract_plain_text(pdf)
        start, end = extract_statement_period(full_text)
        year = (start or end or datetime.today().date()).year
        company = guess_company_name(full_text)

        blob = full_text.lower()
        is_indep    = ('independent bank' in blob) or ('checks in number order' in blob and 'daily balance summary' in blob and 'deposits and other credits' in blob)
        is_banner   = ("banner bank" in blob) or ("banner's best checking" in blob) or ("banners best checking" in blob) or ("checks in number order" in blob and "daily balance summary" in blob)
        is_regions  = ('regions bank' in blob) or ('regions.com' in blob) or ('deposits & credits' in blob and 'daily balance summary' in blob)
        is_investors= ('investors bank' in blob) or ('investorsbank.com' in blob) or ('citizens bank, n.a.' in blob) or ('account activity detail' in blob and 'check register' in blob)
        is_ecsb     = ('ecsb.com' in blob) or ('east cambridge savings' in blob) or ('no hidden fees business check' in blob)
        is_keybank  = ('keybank' in blob) or ('key.com' in blob) or ('additions' in blob and 'subtractions' in blob)
        is_bofa     = ('bank of america' in blob) or ('bankofamerica.com' in blob) or ('daily ledger balances' in blob)
        is_chase    = ('jpmorgan chase' in blob) or ('chase business' in blob) or ('chase.com' in blob)

        if is_indep:
            txns, daily = parse_independent(full_text, period_year=year)
        elif is_banner:
            txns, daily = parse_banner(full_text, period_year=year)  # defined below
        elif is_regions:
            txns, daily = parse_regions(full_text, period_year=year)
        elif is_investors:
            txns, daily = parse_investors(full_text, period_year=year)
        elif is_ecsb:
            tables = extract_tables(pdf)
            txns, daily = parse_ecsb_tables(tables, year)
            if not daily:
                try:
                    df = pd.DataFrame(txns)
                    if 'balance' in df and df['balance'].notna().any():
                        s = df[['date','balance']].dropna().groupby('date')['balance'].last()
                        daily = {k: float(v) for k, v in s.to_dict().items()}
                except Exception:
                    daily = None
        elif is_keybank:
            txns, daily = parse_keybank(full_text, period_year=year)
        elif is_bofa:
            txns, daily = parse_bofa(full_text, period_year=year)
        elif is_chase:
            txns, daily = parse_chase(full_text, period_year=year)
        else:
            tables = extract_tables(pdf)
            txns = coerce_transactions_from_tables(tables= tables)
            if not txns:
                txns = coerce_transactions_from_text(full_text)
            daily = None

    for t in txns:
        t['source_file'] = os.path.basename(path)

    summary = summarize(txns, cfg, daily_balances=daily)
    period_label = ""
    if start and end:
        period_label = f"{start} to {end}"
    elif start:
        period_label = f"{start}"
    elif end:
        period_label = f"up to {end}"

    meta = {'Company Name': company, 'Month': period_label or ""}
    meta.update(summary)
    return txns, meta

# ------------------ Banner parser (used above) ------------------
BANNER_HEADERS = [
    "Deposits and Additions",
    "Withdrawals and Deductions",
    "Checks In Number Order",
    "Daily Balance Summary"
]
BANNER_CHECK_TWOCOL = re.compile(r'(\d{1,2}/\d{2})\s+(\d{3,}\*?)\s+\$?([0-9,]+\.\d{2})')

def parse_banner_block(block_text, year, sign, section_name):
    if not block_text:
        return []
    lines = [l.rstrip() for l in block_text.splitlines()]
    if lines:
        lines = lines[1:] if lines[0].strip() else lines
    txns, current = [], []
    def is_total(s): return s.strip().lower().startswith("total")
    def flush(block):
        if not block: return None
        first = block[0]
        if is_total(first): return None
        m = MD_SLASH.match(first)
        if not m: return None
        md = m.group(1)
        d = try_parse_date(md, default_year=year)
        if not d: return None
        text = " ".join([b.strip() for b in block if b.strip() and not is_total(b)])
        amt = parse_amount_from_block(text)
        if amt is None: return None
        desc = text[len(md):].strip(" -:•")
        return {'date': d.date(), 'description': desc, 'amount': amt*sign, 'balance': None, 'section': section_name}
    for line in lines:
        if not line.strip(): continue
        if is_total(line):
            if current:
                rec = flush(current)
                if rec: txns.append(rec)
                current = []
            continue
        if MD_SLASH.match(line):
            if current:
                rec = flush(current)
                if rec: txns.append(rec)
                current = []
            current.append(line)
        else:
            if current: current.append(line)
    if current:
        rec = flush(current)
        if rec: txns.append(rec)
    return txns

def parse_banner_checks(checks_text, year):
    if not checks_text:
        return []
    lines = [l.strip() for l in checks_text.splitlines()]
    out = []
    for line in lines:
        low = line.lower()
        if low.startswith("date") or low.startswith("checks in") or low.startswith("total"):
            continue
        matches = BANNER_CHECK_TWOCOL.findall(line)
        for md, chk, amt in matches:
            d = try_parse_date(md, default_year=year)
            if not d:
                continue
            try:
                amount = float(amt.replace(',', ''))
            except Exception:
                continue
            out.append({'date': d.date(), 'description': f"Check {chk}".replace('*',''), 'amount': -amount, 'balance': None, 'section': 'CHECKS'})
    return out

def parse_banner_daily_balances(section_text, year):
    daily = {}
    if not section_text:
        return daily
    for md, bal in re.findall(r'(\d{2}/\d{2})\s+\$?([0-9,]+\.\d{2})', section_text):
        d = try_parse_date(md, default_year=year)
        if not d:
            continue
        try:
            daily[d.date()] = float(bal.replace(',', ''))
        except Exception:
            pass
    return daily

def parse_banner(full_text, period_year):
    def sec(h):
        nxt = [x for x in BANNER_HEADERS if x != h]
        return find_section(full_text, h, nxt)

    deposits = sec("Deposits and Additions")
    withdrawals = sec("Withdrawals and Deductions")
    checks = sec("Checks In Number Order")
    daily = sec("Daily Balance Summary")

    txns = []
    txns += parse_banner_block(deposits, period_year, +1, "Deposits and Additions")
    txns += parse_banner_block(withdrawals, period_year, -1, "Withdrawals and Deductions")
    txns += parse_banner_checks(checks, period_year)
    daily_bal = parse_banner_daily_balances(daily, period_year)
    return txns, daily_bal

# ------------------ CLI ------------------
def main():
    ap = argparse.ArgumentParser(description="Bank Statement Analyzer (Independent + Banner + Regions + Investors/Citizens + ECSB + KeyBank + BofA + Chase + generic)")
    ap.add_argument("--input", "-i", required=True, help="Path to a PDF file OR a folder of PDFs")
    ap.add_argument("--config", "-c", default="config.json", help="Path to config.json")
    ap.add_argument("--outdir", "-o", default="output", help="Output folder")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.outdir, exist_ok=True)

    files = []
    if os.path.isdir(args.input):
        for name in os.listdir(args.input):
            if name.lower().endswith(".pdf"):
                files.append(os.path.join(args.input, name))
    elif args.input.lower().endswith(".pdf"):
        files.append(args.input)
    else:
        print("Input must be a PDF file or a folder containing PDFs.")
        sys.exit(1)

    all_txns = []
    summaries = []

    for f in sorted(files):
        try:
            txns, meta = process_pdf(f, cfg)
            all_txns.extend(txns)
            summaries.append(meta)
            print(f"[OK] {os.path.basename(f)} -> {meta['Company Name']} | {meta['Month']}")
        except Exception as e:
            print(f"[ERR] {os.path.basename(f)}: {e}")

    if all_txns:
        txdf = pd.DataFrame(all_txns)
        cols = [c for c in ['source_file','date','description','amount','balance','section'] if c in txdf.columns]
        txdf = txdf[cols]
        txdf.sort_values(by=['source_file', 'date'], inplace=True)
        txdf.to_csv(os.path.join(args.outdir, "transactions.csv"), index=False)

    if summaries:
        cols = [
            'Company Name', 'Month',
            'Total Deposits', 'Number of Deposits',
            'Number of Negative Days', 'Avg Daily Balance',
            'Number of Cash Advances', 'MCA Company Names',
            'Payment Amount', 'Payment Frequency',
            'Amount Changed Over 3 Months',
            'MCA Fundings (Count)', 'MCA Fundings (Total)', 'MCA Funding (Largest)',
            'Large Deposits (Count)', 'Large Deposits (Total)', 'Largest Deposit (Any)'
        ]
        sdf = pd.DataFrame(summaries, columns=cols)
        sdf.to_excel(os.path.join(args.outdir, "summary.xlsx"), index=False)

    print(f"\nDone. Outputs in: {os.path.abspath(args.outdir)}")
    if not all_txns and not summaries:
        print("No data extracted. Share another sample and I’ll refine the parser for that bank.")

if __name__ == "__main__":
    main()
