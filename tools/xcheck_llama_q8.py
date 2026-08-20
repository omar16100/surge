#!/usr/bin/env python3
"""M3.4 gate (B): surge Q8_0 greedy vs llama.cpp greedy, same 27B Q8_0 GGUF.

    /Users/macmini/models/dsv4-venv/bin/python tools/xcheck_llama_q8.py \
        --gguf /Users/macmini/models/gguf/Qwen3.6-27B-Q8_0.gguf

Gate A compares two SURGE paths (Metal and CPU-ref) that read the same Q8_0
bytes, so a bug shared by both would pass it. This gate adds an INDEPENDENT
oracle: llama.cpp decoding the identical GGUF. For each of a few short raw
prompts it runs both engines greedily (temp 0, argmax) for -n tokens and
compares the generated completions.

Tokenizer parity is a precondition and is checked first: surge tokenizes raw
via its GGUF BPE (add_bos_token=false in this GGUF), and llama-tokenize must
produce the identical prompt id list -- otherwise the two engines are not
being fed the same sequence and any divergence is meaningless.

llama.cpp side uses `llama-simple` (minimal greedy example: no chat template,
no jinja, no thinking wrapper, respects add_bos), not the interactive
`llama-cli`. surge side uses `surge -p ... -n N`.

Expectation: the completions agree for the bulk of tokens. A first divergence
at a legitimate near-tie is acceptable and is documented (position + surge's
top1-top2 margin there). An EARLY systematic divergence (differs within the
first few tokens on multiple prompts) means surge's Q8_0 forward disagrees
with llama.cpp and must be investigated, not passed.
"""
import argparse
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_GGUF = "/Users/macmini/models/gguf/Qwen3.6-27B-Q8_0.gguf"

PROMPTS = [
    "The capital of France is",
    "Water is made of two atoms of hydrogen and one atom of",
    "The first four prime numbers are 2, 3, 5, and",
    "In 1969, the Apollo 11 mission landed the first humans on the",
]


def run(cmd, **kw):
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return p.returncode, p.stdout, p.stderr


def surge_greedy(surge, gguf, prompt, n):
    """(prompt_ids, gen_ids, completion_text, margins) from `surge -p ... -n N`."""
    rc, out, err = run([str(surge), str(gguf), "-p", prompt, "-n", str(n), "--margins"])
    if rc != 0:
        raise SystemExit(f"surge failed: {err}")
    prompt_ids, gen_ids, completion = [], [], ""
    for line in out.splitlines():
        if line.startswith("prompt_ids:"):
            prompt_ids = [int(x) for x in line.split(":", 1)[1].strip().split(",") if x]
        elif line.startswith("gen_ids:"):
            gen_ids = [int(x) for x in line.split(":", 1)[1].strip().split(",") if x]
    # `text: ` is printed last on stdout and its value may contain newlines,
    # so take everything after it rather than a single line.
    if "text: " in out:
        completion = out.split("text: ", 1)[1]
        if completion.endswith("\n"):
            completion = completion[:-1]
    # --margins prints "  margin[i] tok <id> gap <g>" on stderr, one per token.
    margins = []
    for line in err.splitlines():
        s = line.strip()
        if s.startswith("margin["):
            try:
                margins.append(float(s.split("gap", 1)[1].strip()))
            except (IndexError, ValueError):
                pass
    return prompt_ids, gen_ids, completion, margins


def llama_tokenize(gguf, prompt):
    rc, out, _ = run(["/opt/homebrew/bin/llama-tokenize", "-m", str(gguf),
                      "-p", prompt, "--log-disable"])
    if rc != 0:
        return None
    ids = []
    for line in out.splitlines():
        # lines look like "   760 -> 'The'"
        head = line.split("->", 1)[0].strip()
        if head.lstrip("-").isdigit():
            ids.append(int(head))
    return ids


def llama_simple(gguf, prompt, n):
    """llama.cpp greedy completion text (prompt echo stripped)."""
    rc, out, err = run(["/opt/homebrew/bin/llama-simple", "-m", str(gguf),
                        "-n", str(n), "-ngl", "999", prompt])
    if rc != 0:
        raise SystemExit(f"llama-simple failed: {err[-2000:]}")
    full = out
    if full.endswith("\n"):
        full = full[:-1]
    if full.startswith(prompt):
        return full[len(prompt):], full
    # llama detokenized the prompt to something other than the input string;
    # fall back to no strip and let the caller see the raw text.
    return None, full


def first_divergence(a, b):
    """Char index of the first differing character (len(shorter) if one is a
    prefix of the other, None if identical)."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) == len(b):
        return None
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--surge", default=str(REPO / "surge"))
    ap.add_argument("-n", "--n-gen", type=int, default=32)
    ap.add_argument("--early", type=int, default=4,
                    help="divergence within this many tokens on >1 prompt = systematic")
    ap.add_argument("--out", default=str(REPO / "tests" / "fixtures" / "m3q8" / "xcheck.json"))
    args = ap.parse_args()

    surge = pathlib.Path(args.surge)
    gguf = pathlib.Path(args.gguf)
    if not surge.exists():
        raise SystemExit(f"{surge} not found; run `make surge`")
    for tool in ("llama-simple", "llama-tokenize"):
        if not pathlib.Path("/opt/homebrew/bin", tool).exists():
            raise SystemExit(f"/opt/homebrew/bin/{tool} not found")

    rows = []
    early_diverge = 0
    tok_parity_ok = True
    print(f"gate B: surge vs llama.cpp greedy, -n {args.n_gen}, {len(PROMPTS)} prompts\n")
    for pi, prompt in enumerate(PROMPTS):
        s_pids, s_gids, s_text, margins = surge_greedy(surge, gguf, prompt, args.n_gen)
        l_pids = llama_tokenize(gguf, prompt)
        parity = (l_pids == s_pids)
        tok_parity_ok = tok_parity_ok and parity
        l_text, l_full = llama_simple(gguf, prompt, args.n_gen)

        div_char = None
        div_note = ""
        if l_text is None:
            div_note = "llama prompt echo did not match input; compared raw"
            div_char = first_divergence(s_text, l_full)
            cmp_text = l_full
        else:
            cmp_text = l_text
            div_char = first_divergence(s_text, cmp_text)

        exact = (div_char is None)
        # Map the char offset to how many surge gen tokens fully agreed, by
        # walking surge's gen tokens is not possible from text alone; instead
        # report the matched-character run and the surge margin at the token
        # that owns the first diverging char (best-effort: proportional).
        matched_chars = len(s_text) if exact else div_char
        approx_tok = None
        if not exact and len(s_text) > 0 and s_gids:
            approx_tok = min(len(s_gids) - 1,
                             int(round(matched_chars / max(1, len(s_text)) * len(s_gids))))
        first_bad_margin = None
        if approx_tok is not None and approx_tok < len(margins):
            first_bad_margin = margins[approx_tok]

        if not exact and approx_tok is not None and approx_tok < args.early:
            early_diverge += 1

        print(f"prompt {pi}: {prompt!r}")
        print(f"  tokenizer parity: {'OK' if parity else 'MISMATCH'} "
              f"(surge {len(s_pids)} tok, llama {len(l_pids) if l_pids else '?'} tok)")
        print(f"  surge gen ({len(s_gids)}): {s_text!r}")
        print(f"  llama gen      : {cmp_text!r}")
        if exact:
            print(f"  => IDENTICAL completion ({len(s_text)} chars)")
        else:
            ctx_a = s_text[max(0, matched_chars - 20):matched_chars + 20]
            ctx_b = cmp_text[max(0, matched_chars - 20):matched_chars + 20]
            print(f"  => diverge at char {matched_chars} (~gen token {approx_tok}"
                  f"{', surge margin %.3e' % first_bad_margin if first_bad_margin is not None else ''})")
            print(f"     surge: ...{ctx_a!r}")
            print(f"     llama: ...{ctx_b!r}")
            if div_note:
                print(f"     note: {div_note}")
        print()

        rows.append({
            "prompt": prompt,
            "tok_parity": parity,
            "surge_prompt_ids": s_pids,
            "llama_prompt_ids": l_pids,
            "surge_gen_ids": s_gids,
            "surge_completion": s_text,
            "llama_completion": cmp_text,
            "identical": exact,
            "diverge_char": None if exact else matched_chars,
            "approx_diverge_token": approx_tok,
            "surge_margin_at_divergence": first_bad_margin,
        })

    systematic = early_diverge >= 2
    n_ident = sum(1 for r in rows if r["identical"])
    print("-" * 64)
    print(f"tokenizer parity: {'OK on all prompts' if tok_parity_ok else 'MISMATCH -> INVALID'}")
    print(f"identical completions: {n_ident}/{len(rows)}")
    print(f"early ( < {args.early} tok) divergences: {early_diverge}")
    verdict = "PASS" if (tok_parity_ok and not systematic) else "FAIL"
    print(f"M3.4 GATE B ({verdict}): "
          f"{'no early systematic divergence' if not systematic else 'EARLY SYSTEMATIC DIVERGENCE'}"
          f"{'' if tok_parity_ok else '; tokenizer parity broken'}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "gguf": str(gguf), "n_gen": args.n_gen,
        "tok_parity_ok": tok_parity_ok,
        "identical": n_ident, "of": len(rows),
        "early_divergences": early_diverge, "systematic": systematic,
        "pass": bool(tok_parity_ok and not systematic),
        "prompts": rows,
    }, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0 if (tok_parity_ok and not systematic) else 1


if __name__ == "__main__":
    sys.exit(main())
