"""
Sürdürülebilirlik Sonuç Matrisi
===============================

Excel çıktısını şu yapıda HTML'e dönüştürür:

    Şirket | Yıl | Soru 1 | Soru 2 | ... | Soru N
    Arçelik| 2023| Cevap  | Cevap  | ... | Cevap

Her hücrede kısa cevap gösterilir. Hücreye tıklandığında tam cevap,
durum, güven, sayfalar ve kaynak metni açılır.

Kullanım:
    pip install pandas openpyxl
    python visualize_results_improved.py

Birden fazla raporu aynı tabloda göstermek için INPUT_FILES listesine
başka Excel yolları ekleyebilirsin.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


INPUT_FILES = [
    Path(
        r"C:\Users\rvdnr\Desktop\sustainability_analyzer\outputs\arcelik_2023_answers.xlsx"
    ),
]

OUTPUT_FILE = Path("sustainability_answer_matrix.html")

QUESTION_HEADER_ALIASES = {
    "soru no",
    "soru numarası",
    "question no",
    "question number",
}

STATUS_COMPLETED = {
    "completed",
    "complete",
    "answered",
    "success",
}


def clean_text(value: Any) -> str:
    """NaN ve boş değerleri güvenli biçimde temizler."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalize_label(value: Any) -> str:
    """Başlık karşılaştırmaları için sade metin üretir."""
    return re.sub(r"\s+", " ", clean_text(value).casefold())


def split_answer_and_sources(answer: str) -> tuple[str, str]:
    """
    Hücrede gösterilecek kısa cevabı kaynaklardan ayırır.

    ExcelExporter cevabın sonuna ``Kaynaklar:`` bölümü ekliyorsa görünür
    tabloda yalnızca cevap kısmı kalır; tam metin detay penceresinde korunur.
    """
    answer = clean_text(answer)
    if not answer:
        return "", ""

    parts = re.split(
        r"\n\s*(?:Kaynaklar|Sources)\s*:\s*\n",
        answer,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    short_answer = parts[0].strip()
    sources = parts[1].strip() if len(parts) > 1 else ""
    return short_answer, sources


def find_question_header_row(raw: pd.DataFrame) -> int:
    """Excel içindeki gerçek soru tablosunun başladığı satırı bulur."""
    for row_index in range(len(raw)):
        first_cell = normalize_label(raw.iat[row_index, 0])
        if first_cell in QUESTION_HEADER_ALIASES:
            return row_index

    raise ValueError(
        "Excel içinde 'Soru No' başlıklı soru tablosu bulunamadı."
    )


def read_metadata(raw: pd.DataFrame, header_row: int) -> dict[str, str]:
    """Soru tablosundan önceki şirket/rapor metadata alanlarını okur."""
    metadata: dict[str, str] = {}

    for row_index in range(header_row):
        key = clean_text(raw.iat[row_index, 0])
        value = clean_text(raw.iat[row_index, 1]) if raw.shape[1] > 1 else ""

        if key:
            metadata[normalize_label(key)] = value

    company = (
        metadata.get("şirket")
        or metadata.get("firma")
        or clean_text(raw.iat[0, 1])
        or "Bilinmeyen şirket"
    )

    year = (
        metadata.get("rapor yılı")
        or metadata.get("yıl")
        or metadata.get("report year")
        or ""
    )

    report_id = (
        metadata.get("rapor kimliği")
        or metadata.get("report id")
        or ""
    )

    report_url = (
        metadata.get("rapor url")
        or metadata.get("report url")
        or ""
    )

    return {
        "company": company,
        "year": year,
        "report_id": report_id,
        "report_url": report_url,
    }


def map_question_columns(headers: list[str]) -> dict[str, int]:
    """Türkçe/İngilizce başlıkları standart alanlara eşler."""
    aliases = {
        "question_no": {"soru no", "question no", "question number"},
        "question": {"soru", "question"},
        "answer": {"cevap", "answer"},
        "status": {"durum", "status"},
        "confidence": {"güven", "confidence"},
        "evidence_ids": {"kanıt kimlikleri", "evidence ids"},
        "pages": {"sayfalar", "pages"},
        "error": {"hata", "error"},
    }

    mapped: dict[str, int] = {}
    normalized_headers = [normalize_label(header) for header in headers]

    for field, candidates in aliases.items():
        for column_index, normalized in enumerate(normalized_headers):
            if normalized in candidates:
                mapped[field] = column_index
                break

    required = {"question_no", "question", "answer", "status"}
    missing = sorted(required - mapped.keys())
    if missing:
        raise ValueError(
            "Soru tablosunda gerekli sütunlar bulunamadı: "
            + ", ".join(missing)
        )

    return mapped


def parse_report(path: Path) -> dict[str, Any]:
    """Bir Excel raporunu soru-cevap matrisi satırına dönüştürür."""
    raw = pd.read_excel(path, header=None, dtype=object)
    header_row = find_question_header_row(raw)
    metadata = read_metadata(raw, header_row)

    headers = [
        clean_text(raw.iat[header_row, column_index])
        for column_index in range(raw.shape[1])
    ]
    column_map = map_question_columns(headers)

    questions: list[dict[str, Any]] = []

    for row_index in range(header_row + 1, len(raw)):
        question_no = clean_text(
            raw.iat[row_index, column_map["question_no"]]
        )
        question = clean_text(raw.iat[row_index, column_map["question"]])

        if not question_no and not question:
            continue

        answer = clean_text(raw.iat[row_index, column_map["answer"]])
        status = clean_text(raw.iat[row_index, column_map["status"]])
        confidence = clean_text(
            raw.iat[row_index, column_map["confidence"]]
        ) if "confidence" in column_map else ""
        evidence_ids = clean_text(
            raw.iat[row_index, column_map["evidence_ids"]]
        ) if "evidence_ids" in column_map else ""
        pages = clean_text(
            raw.iat[row_index, column_map["pages"]]
        ) if "pages" in column_map else ""
        error = clean_text(
            raw.iat[row_index, column_map["error"]]
        ) if "error" in column_map else ""

        short_answer, sources = split_answer_and_sources(answer)

        questions.append(
            {
                "number": question_no,
                "question": question,
                "answer": short_answer,
                "full_answer": answer,
                "sources": sources,
                "status": status,
                "confidence": confidence,
                "evidence_ids": evidence_ids,
                "pages": pages,
                "error": error,
            }
        )

    return {
        **metadata,
        "source_file": str(path),
        "questions": questions,
    }


def build_payload(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Tüm şirket ve yıllar için ortak soru sütunlarını oluşturur."""
    question_registry: dict[str, dict[str, str]] = {}

    for report in reports:
        for item in report["questions"]:
            key = item["number"] or item["question"]
            question_registry.setdefault(
                key,
                {
                    "key": key,
                    "number": item["number"],
                    "question": item["question"],
                },
            )

    def question_sort_key(item: dict[str, str]) -> tuple[int, str]:
        number = item["number"]
        try:
            return int(float(number)), item["question"]
        except (TypeError, ValueError):
            return 10_000, number or item["question"]

    columns = sorted(question_registry.values(), key=question_sort_key)

    rows = []
    for report in reports:
        answer_map = {
            (item["number"] or item["question"]): item
            for item in report["questions"]
        }

        completed_count = sum(
            normalize_label(item["status"]) in STATUS_COMPLETED
            for item in report["questions"]
        )
        total_count = len(report["questions"])
        completion = (
            round(100 * completed_count / total_count)
            if total_count
            else 0
        )

        rows.append(
            {
                "company": report["company"],
                "year": report["year"],
                "report_id": report["report_id"],
                "report_url": report["report_url"],
                "completion": completion,
                "completed_count": completed_count,
                "total_count": total_count,
                "answers": answer_map,
            }
        )

    return {
        "columns": columns,
        "rows": rows,
    }


def build_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False).replace(
        "</", "<\\/"
    )
    return HTML_TEMPLATE.replace("__DATA_JSON__", data_json)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sürdürülebilirlik Cevap Matrisi</title>
<style>
:root {
  --bg: #f4f6f3;
  --panel: #ffffff;
  --ink: #182019;
  --soft: #657066;
  --line: #dfe5de;
  --accent: #315b3b;
  --accent-soft: #e8f1ea;
  --ok: #1f6b39;
  --ok-bg: #e7f4ea;
  --missing: #8a6424;
  --missing-bg: #fff5df;
  --error: #9a3535;
  --error-bg: #faeaea;
  --sans: Inter, "Segoe UI", Arial, sans-serif;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  padding: 26px;
  background: var(--bg);
  color: var(--ink);
  font-family: var(--sans);
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 20px;
  margin-bottom: 18px;
  flex-wrap: wrap;
}

h1 {
  margin: 0;
  font-size: 23px;
  letter-spacing: -0.02em;
}

.subtitle {
  margin-top: 6px;
  color: var(--soft);
  font-size: 13px;
}

.controls {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

input, select {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  color: var(--ink);
  padding: 9px 12px;
  font-size: 13px;
  outline: none;
}

input:focus, select:focus {
  border-color: var(--accent);
}

.summary {
  display: flex;
  gap: 10px;
  margin-bottom: 14px;
  flex-wrap: wrap;
}

.summary-card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 9px;
  padding: 10px 13px;
  font-size: 12px;
  color: var(--soft);
}

.summary-card strong {
  display: block;
  color: var(--ink);
  font-size: 17px;
  margin-bottom: 2px;
}

.table-wrap {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  overflow: auto;
  max-height: 76vh;
  box-shadow: 0 8px 24px rgba(20, 35, 23, 0.05);
}

table {
  border-collapse: separate;
  border-spacing: 0;
  min-width: 100%;
}

th, td {
  border-right: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
}

thead th {
  position: sticky;
  top: 0;
  z-index: 3;
  min-width: 250px;
  max-width: 310px;
  padding: 12px;
  background: var(--accent);
  color: white;
  text-align: left;
  vertical-align: top;
}

thead th .question-no {
  display: inline-block;
  margin-bottom: 6px;
  padding: 2px 7px;
  border-radius: 999px;
  background: rgba(255,255,255,0.17);
  font-size: 11px;
}

thead th .question-text {
  display: block;
  font-size: 12px;
  line-height: 1.35;
  font-weight: 600;
}

th.company-col, td.company-col {
  position: sticky;
  left: 0;
  z-index: 4;
  min-width: 190px;
  max-width: 190px;
}

th.year-col, td.year-col {
  position: sticky;
  left: 190px;
  z-index: 4;
  min-width: 82px;
  max-width: 82px;
}

thead th.company-col,
thead th.year-col {
  z-index: 6;
}

tbody td.company-col,
tbody td.year-col {
  background: var(--panel);
}

td {
  min-width: 250px;
  max-width: 310px;
  padding: 10px;
  vertical-align: top;
  background: var(--panel);
}

tbody tr:hover td {
  background: #fafcf9;
}

tbody tr:hover td.company-col,
tbody tr:hover td.year-col {
  background: #f3f7f2;
}

.company-name {
  font-weight: 700;
  font-size: 13px;
}

.report-id {
  margin-top: 4px;
  color: var(--soft);
  font-size: 10px;
  overflow-wrap: anywhere;
}

.completion {
  margin-top: 9px;
  display: flex;
  align-items: center;
  gap: 7px;
  color: var(--soft);
  font-size: 11px;
}

.progress {
  width: 72px;
  height: 6px;
  border-radius: 999px;
  overflow: hidden;
  background: var(--line);
}

.progress span {
  display: block;
  height: 100%;
  background: var(--accent);
}

.answer-card {
  cursor: pointer;
  border-radius: 8px;
  padding: 9px;
  min-height: 92px;
  transition: transform 0.12s ease, box-shadow 0.12s ease;
}

.answer-card:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 14px rgba(26, 45, 29, 0.09);
}

.answer-card.completed {
  background: var(--ok-bg);
}

.answer-card.insufficient {
  background: var(--missing-bg);
}

.answer-card.failed {
  background: var(--error-bg);
}

.answer-card.empty {
  background: #f4f5f3;
  color: var(--soft);
}

.badge {
  display: inline-block;
  margin-bottom: 7px;
  border-radius: 999px;
  padding: 3px 7px;
  font-size: 10px;
  font-weight: 700;
}

.completed .badge {
  color: var(--ok);
  background: rgba(255,255,255,0.65);
}

.insufficient .badge {
  color: var(--missing);
  background: rgba(255,255,255,0.65);
}

.failed .badge {
  color: var(--error);
  background: rgba(255,255,255,0.65);
}

.answer-text {
  display: -webkit-box;
  -webkit-line-clamp: 5;
  -webkit-box-orient: vertical;
  overflow: hidden;
  white-space: normal;
  overflow-wrap: anywhere;
  line-height: 1.42;
  font-size: 12px;
}

.cell-meta {
  margin-top: 8px;
  color: var(--soft);
  font-size: 10px;
}

.modal {
  display: none;
  position: fixed;
  inset: 0;
  z-index: 20;
  background: rgba(10, 18, 12, 0.52);
  padding: 30px;
}

.modal.open {
  display: flex;
  align-items: center;
  justify-content: center;
}

.modal-panel {
  width: min(860px, 96vw);
  max-height: 88vh;
  overflow: auto;
  background: var(--panel);
  border-radius: 14px;
  padding: 22px;
  box-shadow: 0 24px 80px rgba(0,0,0,0.28);
}

.modal-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 18px;
}

.modal h2 {
  margin: 0;
  font-size: 18px;
  line-height: 1.35;
}

.close {
  border: 0;
  background: #edf0ec;
  border-radius: 8px;
  padding: 7px 10px;
  cursor: pointer;
  font-size: 16px;
}

.detail-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(120px, 1fr));
  gap: 9px;
  margin: 16px 0;
}

.detail-item {
  padding: 9px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fafbf9;
  font-size: 11px;
}

.detail-item strong {
  display: block;
  margin-bottom: 4px;
  color: var(--soft);
}

.detail-section {
  margin-top: 18px;
}

.detail-section h3 {
  margin: 0 0 8px;
  font-size: 13px;
}

.detail-text {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  line-height: 1.55;
  font-size: 13px;
}

@media (max-width: 760px) {
  body { padding: 14px; }
  .detail-grid { grid-template-columns: 1fr 1fr; }
}
</style>
</head>
<body>

<div class="page-header">
  <div>
    <h1>Sürdürülebilirlik Cevap Matrisi</h1>
    <div class="subtitle">
      Her sütun bir soruyu, her satır bir şirket-yıl raporunu temsil eder.
      Hücreye tıklayarak ayrıntıları açabilirsin.
    </div>
  </div>
  <div class="controls">
    <input id="company-search" type="search" placeholder="Şirket ara">
    <select id="status-filter">
      <option value="">Tüm cevaplar</option>
      <option value="completed">Cevaplananlar</option>
      <option value="insufficient">Kanıt yetersiz</option>
      <option value="failed">Hatalılar</option>
    </select>
  </div>
</div>

<div class="summary">
  <div class="summary-card">
    <strong id="company-count">0</strong>
    şirket-yıl
  </div>
  <div class="summary-card">
    <strong id="question-count">0</strong>
    soru
  </div>
  <div class="summary-card">
    <strong id="answered-count">0</strong>
    cevaplanan hücre
  </div>
</div>

<div class="table-wrap">
  <table>
    <thead>
      <tr id="header-row"></tr>
    </thead>
    <tbody id="body"></tbody>
  </table>
</div>

<div class="modal" id="detail-modal">
  <div class="modal-panel">
    <div class="modal-head">
      <h2 id="modal-question"></h2>
      <button class="close" id="modal-close" aria-label="Kapat">✕</button>
    </div>

    <div class="detail-grid">
      <div class="detail-item"><strong>Şirket</strong><span id="modal-company"></span></div>
      <div class="detail-item"><strong>Yıl</strong><span id="modal-year"></span></div>
      <div class="detail-item"><strong>Durum</strong><span id="modal-status"></span></div>
      <div class="detail-item"><strong>Güven</strong><span id="modal-confidence"></span></div>
      <div class="detail-item"><strong>Sayfalar</strong><span id="modal-pages"></span></div>
      <div class="detail-item"><strong>Kanıtlar</strong><span id="modal-evidence"></span></div>
    </div>

    <div class="detail-section">
      <h3>Cevap</h3>
      <div class="detail-text" id="modal-answer"></div>
    </div>

    <div class="detail-section" id="sources-section">
      <h3>Kaynak metinleri</h3>
      <div class="detail-text" id="modal-sources"></div>
    </div>

    <div class="detail-section" id="error-section">
      <h3>Hata</h3>
      <div class="detail-text" id="modal-error"></div>
    </div>
  </div>
</div>

<script>
const DATA = __DATA_JSON__;

function normalizeStatus(status) {
  const value = (status || '').trim().toLowerCase();

  if (['completed', 'complete', 'answered', 'success'].includes(value)) {
    return 'completed';
  }
  if (value.includes('insufficient')) {
    return 'insufficient';
  }
  if (value.includes('fail') || value.includes('error')) {
    return 'failed';
  }
  return value ? 'other' : 'empty';
}

function statusLabel(statusClass, rawStatus) {
  if (statusClass === 'completed') return 'Cevaplandı';
  if (statusClass === 'insufficient') return 'Kanıt yetersiz';
  if (statusClass === 'failed') return 'Hata';
  if (statusClass === 'empty') return 'Cevap yok';
  return rawStatus || 'Belirsiz';
}

function renderHeader() {
  const row = document.getElementById('header-row');
  row.innerHTML = '';

  const company = document.createElement('th');
  company.className = 'company-col';
  company.textContent = 'Şirket';
  row.appendChild(company);

  const year = document.createElement('th');
  year.className = 'year-col';
  year.textContent = 'Yıl';
  row.appendChild(year);

  DATA.columns.forEach(column => {
    const th = document.createElement('th');
    th.innerHTML = `
      <span class="question-no">Soru ${escapeHtml(column.number || column.key)}</span>
      <span class="question-text">${escapeHtml(column.question)}</span>
    `;
    row.appendChild(th);
  });
}

function renderBody() {
  const query = document.getElementById('company-search')
    .value.trim().toLowerCase();
  const statusFilter = document.getElementById('status-filter').value;

  const rows = DATA.rows.filter(row =>
    `${row.company} ${row.year} ${row.report_id}`
      .toLowerCase()
      .includes(query)
  );

  const body = document.getElementById('body');
  body.innerHTML = '';

  let visibleAnswered = 0;

  rows.forEach(row => {
    const tr = document.createElement('tr');

    const companyCell = document.createElement('td');
    companyCell.className = 'company-col';
    companyCell.innerHTML = `
      <div class="company-name">${escapeHtml(row.company)}</div>
      <div class="report-id">${escapeHtml(row.report_id || '')}</div>
      <div class="completion">
        <div class="progress"><span style="width:${row.completion}%"></span></div>
        <span>${row.completed_count}/${row.total_count}</span>
      </div>
    `;
    tr.appendChild(companyCell);

    const yearCell = document.createElement('td');
    yearCell.className = 'year-col';
    yearCell.textContent = row.year || '—';
    tr.appendChild(yearCell);

    DATA.columns.forEach(column => {
      const item = row.answers[column.key] || {};
      const statusClass = normalizeStatus(item.status);
      if (statusClass === 'completed') visibleAnswered += 1;

      const td = document.createElement('td');
      const card = document.createElement('div');
      card.className = `answer-card ${statusClass}`;

      const shouldDim = statusFilter && statusClass !== statusFilter;
      if (shouldDim) {
        card.style.opacity = '0.18';
      }

      const answer = item.answer || (
        statusClass === 'empty'
          ? 'Bu rapor için cevap bulunamadı.'
          : 'Açıklama bulunamadı.'
      );

      card.innerHTML = `
        <span class="badge">${escapeHtml(statusLabel(statusClass, item.status))}</span>
        <div class="answer-text">${escapeHtml(answer)}</div>
        <div class="cell-meta">
          ${item.pages ? `Sayfa: ${escapeHtml(item.pages)}` : ''}
          ${item.confidence ? ` · Güven: ${escapeHtml(item.confidence)}` : ''}
        </div>
      `;

      card.onclick = () => openModal(row, column, item);
      td.appendChild(card);
      tr.appendChild(td);
    });

    body.appendChild(tr);
  });

  document.getElementById('company-count').textContent = rows.length;
  document.getElementById('question-count').textContent = DATA.columns.length;
  document.getElementById('answered-count').textContent = visibleAnswered;
}

function openModal(row, column, item) {
  document.getElementById('modal-question').textContent =
    `Soru ${column.number || column.key}: ${column.question}`;
  document.getElementById('modal-company').textContent = row.company || '—';
  document.getElementById('modal-year').textContent = row.year || '—';
  document.getElementById('modal-status').textContent = item.status || '—';
  document.getElementById('modal-confidence').textContent =
    item.confidence || '—';
  document.getElementById('modal-pages').textContent = item.pages || '—';
  document.getElementById('modal-evidence').textContent =
    item.evidence_ids || '—';
  document.getElementById('modal-answer').textContent =
    item.answer || 'Cevap bulunamadı.';
  document.getElementById('modal-sources').textContent =
    item.sources || 'Kaynak metni bulunmuyor.';
  document.getElementById('modal-error').textContent =
    item.error || 'Hata bulunmuyor.';

  document.getElementById('sources-section').style.display =
    item.sources ? 'block' : 'none';
  document.getElementById('error-section').style.display =
    item.error ? 'block' : 'none';

  document.getElementById('detail-modal').classList.add('open');
}

function closeModal() {
  document.getElementById('detail-modal').classList.remove('open');
}

function escapeHtml(value) {
  const div = document.createElement('div');
  div.textContent = value == null ? '' : String(value);
  return div.innerHTML;
}

document.getElementById('company-search')
  .addEventListener('input', renderBody);
document.getElementById('status-filter')
  .addEventListener('change', renderBody);
document.getElementById('modal-close')
  .addEventListener('click', closeModal);
document.getElementById('detail-modal')
  .addEventListener('click', event => {
    if (event.target.id === 'detail-modal') closeModal();
  });
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') closeModal();
});

renderHeader();
renderBody();
</script>
</body>
</html>
"""


def main() -> None:
    missing_files = [path for path in INPUT_FILES if not path.exists()]
    if missing_files:
        missing_text = "\n".join(f"- {path}" for path in missing_files)
        raise FileNotFoundError(
            "Aşağıdaki Excel dosyaları bulunamadı:\n" + missing_text
        )

    reports = [parse_report(path) for path in INPUT_FILES]
    payload = build_payload(reports)
    output_html = build_html(payload)

    OUTPUT_FILE.write_text(output_html, encoding="utf-8")

    print(f"Oluşturuldu: {OUTPUT_FILE.resolve()}")
    print(
        f"Şirket-yıl: {len(payload['rows'])} | "
        f"Soru: {len(payload['columns'])}"
    )


if __name__ == "__main__":
    main()