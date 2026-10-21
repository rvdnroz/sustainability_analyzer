from __future__ import annotations

import re
from pathlib import Path


TARGET = Path("retrieval/table_retriever.py")


def replace_required(
    text: str,
    pattern: str,
    replacement: str,
    label: str,
    *,
    flags: int = 0,
) -> str:
    updated, count = re.subn(
        pattern,
        replacement,
        text,
        count=1,
        flags=flags,
    )
    if count != 1:
        raise RuntimeError(
            f"{label} bloğu bulunamadı; dosya sürümü beklenenden farklı."
        )
    return updated


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")

    text = replace_required(
        text,
        r"maximum_row_results_per_table:\s*int\s*=\s*\d+",
        "maximum_row_results_per_table: int = 8",
        "maximum_row_results_per_table",
    )

    text = replace_required(
        text,
        r'("row_values":\s*tuple\(row_values\),\s*\n)(\s*\},)',
        r'''\1                        "direct_metric_match": self._direct_metric_match(
                            query.metric_terms,
                            row_text,
                        ),
\2''',
        "row metadata",
        flags=re.MULTILINE,
    )

    text = replace_required(
        text,
        r'''results\.sort\(\s*
            key=lambda result:\s*-result\.score,\s*
        \)''',
        '''results.sort(
            key=lambda result: (
                not bool(
                    result.metadata.get("direct_metric_match")
                ),
                -result.score,
            ),
        )''',
        "row result sıralaması",
        flags=re.MULTILINE | re.VERBOSE,
    )

    text = replace_required(
        text,
        r'''deduplicated\.sort\(\s*
            key=lambda result:\s*\(\s*
                -result\.score,''',
        '''deduplicated.sort(
            key=lambda result: (
                not bool(
                    result.metadata.get("direct_metric_match")
                ),
                result.metadata.get("result_level") != "row",
                -result.score,''',
        "final result sıralaması",
        flags=re.MULTILINE | re.VERBOSE,
    )

    marker = "    @classmethod\n    def _collection_match_score("
    if marker not in text:
        raise RuntimeError("_collection_match_score marker bulunamadı.")

    helper = '''    @classmethod
    def _direct_metric_match(
        cls,
        metric_terms: Sequence[str],
        row_text: str,
    ) -> bool:
        normalized_row = cls.normalize_text(row_text)
        if not normalized_row:
            return False

        row_tokens = set(cls._tokenize(normalized_row))

        for term in metric_terms:
            normalized_term = cls.normalize_text(str(term))
            if not normalized_term:
                continue

            if len(normalized_term) <= 6:
                if normalized_term in row_tokens:
                    return True
                continue

            if normalized_term in normalized_row:
                return True

        return False

'''
    text = text.replace(marker, helper + marker, 1)

    text = replace_required(
        text,
        r'(?m)^(\s*)return " "\.join\(normalized\.split\(\)\)\s*$',
        r'''\1normalized = " ".join(normalized.split())

\1abbreviation_patterns = {
\1    r"\bno\s+x\b": "nox",
\1    r"\bso\s+x\b": "sox",
\1    r"\bco\s+2\b": "co2",
\1}
\1for pattern, replacement in abbreviation_patterns.items():
\1    normalized = re.sub(
\1        pattern,
\1        replacement,
\1        normalized,
\1    )

\1return normalized''',
        "normalize_text dönüşü",
    )

    TARGET.write_text(text, encoding="utf-8")
    compile(text, str(TARGET), "exec")

    print(f"Güncellendi: {TARGET}")
    print("Ek LLM çağrısı eklenmedi.")
    print("Top-k değişmedi.")
    print("Doğrudan metrik eşleşen tablo satırları öne alındı.")


if __name__ == "__main__":
    main()