"""Solution identity preserves order, multiplicity, parameter settings and conditions."""
import hashlib
import json
import re
from difflib import SequenceMatcher


def ordered_protocol(protocol):
    return [re.sub(r'\s+', '', re.sub(r'<[^>]*>', '<?>', str(p))).lower() for p in protocol]


def solution_contract(skill):
    contract = (skill.metadata or {}).get('execution_contract') or {}
    return {'protocol': ordered_protocol(skill.action_protocol),
            'conditions': sorted(contract.get('observed_conditions') or []),
            'bindings': contract.get('bindings') or {},
            'verification': sorted(contract.get('verification_tools') or []),
            'domain': skill.domain}


def solution_fingerprint(skill):
    return hashlib.sha256(json.dumps(solution_contract(skill), sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]


def solution_distance(left, right):
    if left.get('domain') and right.get('domain') and left['domain'] != right['domain']:
        return 1.0
    seq = 1 - SequenceMatcher(None, left.get('protocol', []), right.get('protocol', []), autojunk=False).ratio()
    def distance(key):
        a, b = left.get(key) or [], right.get(key) or []
        if isinstance(a, dict): a = [json.dumps([k,v], sort_keys=True) for k,v in a.items()]
        if isinstance(b, dict): b = [json.dumps([k,v], sort_keys=True) for k,v in b.items()]
        a, b = set(a), set(b)
        return 1 - len(a & b)/len(a | b) if a or b else 0.0
    # A condition or target-binding change may define a new solution even
    # when the tool names and ordering are identical.
    return max(seq, distance('conditions'), distance('bindings'), distance('verification'))
