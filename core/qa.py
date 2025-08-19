from __future__ import annotations
import re
from typing import Optional

# Very simple pattern-based intent detection.
# For production, replace with an LLM-backed parser that maps NL -> SQL/chart intents.

def parse_intent(q: str):
    q = q.lower().strip()
    # patterns
    m = re.match(r'distribution of (.+)', q)
    if m:
        return {'type':'distribution', 'col': m.group(1).strip()}
    m = re.match(r'(relation|relationship|correlation) between (.+) and (.+)', q)
    if m:
        return {'type':'relationship', 'x': m.group(2).strip(), 'y': m.group(3).strip()}
    m = re.match(r'average of (.+) by (.+)', q)
    if m:
        return {'type':'sql', 'sql': f"SELECT {m.group(2)} AS group_by, AVG({m.group(1)}) AS avg_value FROM t GROUP BY 1 ORDER BY 2 DESC"}
    # fallback -> quick stats
    if 'summary' in q or 'describe' in q or 'stats' in q:
        return {'type':'stats'}
    return {'type':'unknown'}

# Stubs for LLM providers (disabled by default)
def llm_to_sql(question: str, schema_hint: str, provider: str='openai') -> Optional[str]:
    return None  # implement with your favorite provider
