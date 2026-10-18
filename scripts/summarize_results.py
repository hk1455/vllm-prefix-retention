#!/usr/bin/env python3
"""Validate paired published summaries and reproduce aggregate result tables."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compare_results import compare


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def aggregate(runs):
    ms = [x['measurement'] for x in runs]
    total = {k: sum(x[k] for x in ms) for k in (
        'requests', 'successful_requests', 'prompt_tokens', 'saved_prefill_tokens',
        'prior_completed_requests', 'repeat_prefix_recompute_tokens', 'wall_seconds')}
    total['token_hit_rate'] = total['saved_prefill_tokens'] / total['prompt_tokens']
    hits = sum(round(x['repeat_full_prefix_hit_rate'] * x['prior_completed_requests']) for x in ms)
    total['repeat_full_prefix_hit_rate'] = hits / total['prior_completed_requests']
    total['ttft_ms_mean'] = sum(x['ttft_ms_mean'] * x['requests'] for x in ms) / total['requests']
    total['requests_per_second'] = total['successful_requests'] / total['wall_seconds']
    return total


def main():
    root = ROOT/'benchmarks'
    groups = {p: [read(root/'published'/f'{p}_s{s}.json') for s in (42, 43, 44)] for p in ('lru', 'slru')}
    for a, b in zip(groups['lru'], groups['slru']):
        compare(a, b)  # Reject incomplete or incompatible runs before aggregation.
    a, b = (aggregate(groups[p]) for p in ('lru', 'slru'))
    print('Published long runs: matching client configurations and workload fingerprints.')
    print('| Metric | LRU | SLRU | Candidate minus baseline |')
    print('|---|---:|---:|---:|')
    for key in ('repeat_prefix_recompute_tokens', 'saved_prefill_tokens', 'token_hit_rate',
                'repeat_full_prefix_hit_rate', 'ttft_ms_mean', 'requests_per_second'):
        av, bv = a[key], b[key]
        delta = f'{100*(bv-av):+.2f} pp' if key.endswith('rate') else f'{100*(bv/av-1):+.2f}%'
        print(f'| {key} | {av:.6f} | {bv:.6f} | {delta} |')
    historical = aggregate([read(root/'history'/f'slru_s{s}.json') for s in (42,43,44)])
    print(f"\nHistorical SLRU mean TTFT: {historical['ttft_ms_mean']:.6f} ms (separate run).")
    aba = [read(root/'aba'/f'{stage}_zipf.json') for stage in ('01_lru','02_slru','03_lru')]
    for baseline in (aba[0],aba[2]):
        compare(baseline,aba[1])
    print('ABA mean TTFT (ms):', ', '.join(f"{x['measurement']['ttft_ms_mean']:.6f}" for x in aba))
    hot = {p: sum(x['top10']['repeat_prefix_recompute_tokens'] for x in groups[p]) for p in groups}
    tail = {p: aggregate(groups[p])['repeat_prefix_recompute_tokens'] - hot[p] for p in groups}
    for name, values in (('P0-P9', hot), ('P10-P99', tail)):
        print(f"{name} repeat recompute: {values['lru']} -> {values['slru']} ({100*(values['slru']/values['lru']-1):+.2f}%)")
    # Verify payload/reference bytes without importing or installing vLLM.
    import hashlib
    manifest = read(ROOT/'manifest.json')
    for group, prefix in (('original_sha256', ROOT/'reference/vllm'), ('payload_sha256',ROOT/'payload')):
        for relative, expected in manifest[group].items():
            actual = hashlib.sha256((prefix/relative).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'Source checksum mismatch: {prefix/relative}')
    print('All reference and payload SHA256 values match the installer manifest.')


if __name__ == '__main__':
    main()
