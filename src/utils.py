"""
utils.print_table(rows, headers=None, title=None)
-------------------------------------------------
Pretty console table without external deps.
`rows`    : list of dicts  (all rows should share same keys)
`headers` : ordered list of column names (optional; default rows[0].keys())
`title`   : optional single-line heading
"""
from typing import List, Dict, Sequence

def print_table(rows: List[Dict], headers: Sequence[str] = None, title: str = None):
    if not rows:
        print("(no data)")
        return
    if headers is None:
        headers = list(rows[0].keys())

    # column widths
    col_w = {h: max(len(str(h)), max(len(str(r[h])) for r in rows)) for h in headers}

    sep = "+".join("-"*(col_w[h]+2) for h in headers)
    def _row(vals):
        return "|".join(f" {str(v).ljust(col_w[h])} " for h, v in zip(headers, vals))

    if title:
        print(title)
    print(sep)
    print(_row(headers))
    print(sep)
    for r in rows:
        print(_row([r[h] for h in headers]))
    print(sep)
