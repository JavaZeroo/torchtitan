# Prints a short summary of exp_h20/run.sh results, meant to be copied back by
# hand. Usage: python exp_h20/analyze.py [OUT]   (default ../k31_h20_out)

import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "..", "k31_h20_out")
AG_SRC = os.environ.get("AG_SRC", os.path.join(REPO, "..", "attention-gym"))
TAGS = {"loss": "loss_metrics/global_avg_loss", "grad_norm": "grad_norm"}


def read(path, default=""):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return default


def last_error(log_path):
    text = re.sub(r"\x1b\[[0-9;]*m", "", read(log_path))
    hits = [
        m.group(0)
        for m in re.finditer(r"\b[A-Za-z]*(Error|Exception): [^\n]*", text)
        if "numa_node" not in m.group(0)
    ]
    return hits[-1][:160] if hits else "-"


def tb_scalars(tb_dir):
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    if not os.path.isdir(tb_dir):
        return None
    subs = sorted(d for d in os.listdir(tb_dir) if os.path.isdir(os.path.join(tb_dir, d)))
    if not subs:
        return None
    acc = EventAccumulator(os.path.join(tb_dir, subs[-1]), size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    out = {}
    for name, tag in TAGS.items():
        if tag in tags:
            out[name] = {s.step: s.value for s in acc.Scalars(tag)}
    return out or None


def env_section():
    print("## env")
    text = read(os.path.join(OUT, "env", "env.txt"), "missing")
    lines = text.splitlines()
    gpus = [l for l in lines[1:] if re.match(r"^\d+,", l)]
    if gpus:
        print(f"gpus: {len(gpus)} x {gpus[0].split(',', 1)[1].strip()}")
    for l in lines:
        if l.startswith(("torch ", "triton ", "torchao ", "attn_gym ", "cutlass ", "flash_attn_interface ")):
            print(" ".join(l.split()[:3])[:120])
    if not gpus:
        print(text[:300])


def unit_section():
    print("## unit")
    xml = os.path.join(OUT, "unit", "unit.xml")
    if not os.path.exists(xml):
        print("not run")
        return
    root = ET.parse(xml).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    a = suite.attrib
    print(f"tests={a.get('tests')} failures={a.get('failures')} errors={a.get('errors')} skipped={a.get('skipped')}")
    bad = [
        tc.get("classname", "") + "::" + tc.get("name", "")
        for tc in suite.iter("testcase")
        if tc.find("failure") is not None or tc.find("error") is not None
    ]
    for b in bad[:8]:
        print("  FAIL", b)


def ci_section():
    print("## ci (h100 suite, real PG)")
    d = os.path.join(OUT, "ci")
    if not os.path.isdir(d):
        print("not run")
        return
    for f in sorted(os.listdir(d)):
        if f.endswith(".exit"):
            t = f[: -len(".exit")]
            code = read(os.path.join(d, f)).strip()
            err = "" if code == "0" else "  " + last_error(os.path.join(d, t + ".log"))
            print(f"{t}: {'PASS' if code == '0' else 'FAIL exit=' + code}{err}")


def compare(base, other):
    steps = sorted(set(base["loss"]) & set(other["loss"]))
    if not steps:
        return "no common steps"
    dl = max(abs(base["loss"][s] - other["loss"][s]) for s in steps)
    s1 = steps[0]
    eq1 = base["loss"][s1] == other["loss"][s1]
    g1 = ""
    if "grad_norm" in base and "grad_norm" in other and s1 in other["grad_norm"]:
        g1 = " gn1_equal=" + str(base["grad_norm"][s1] == other["grad_norm"][s1])
    last = steps[-1]
    return (
        f"steps={len(steps)} step1_loss_equal={eq1}{g1} "
        f"max|dloss|={dl:.2e} last_loss={other['loss'][last]:.5f}"
    )


def par_section():
    for group, base_name in (("par_p", "p_fsdp2_run1"), ("par_t", "t_fsdp2_run1")):
        print(f"## {group} (vs {base_name}, nondeterministic)")
        g = os.path.join(OUT, group)
        if not os.path.isdir(g):
            print("not run")
            continue
        base = tb_scalars(os.path.join(g, "tb_" + base_name))
        runs = sorted(f[: -len(".exit")] for f in os.listdir(g) if f.endswith(".exit"))
        for r in runs:
            code = read(os.path.join(g, r + ".exit")).strip()
            if code != "0":
                print(f"{r}: FAIL exit={code}  {last_error(os.path.join(g, r + '.log'))}")
                continue
            if r.endswith("_seed"):
                print(f"{r}: ok")
                continue
            m = tb_scalars(os.path.join(g, "tb_" + r))
            if m is None or base is None or r == base_name:
                print(f"{r}: ok{' (baseline)' if r == base_name else ' (no tb)'}")
                continue
            print(f"{r}: {compare(base, m)}")


def neg_section():
    print("## neg (expected to fail)")
    g = os.path.join(OUT, "neg")
    if not os.path.isdir(g):
        print("not run")
        return
    for f in sorted(os.listdir(g)):
        if f.endswith(".exit"):
            r = f[: -len(".exit")]
            code = read(os.path.join(g, f)).strip()
            print(f"{r}: exit={code}  {last_error(os.path.join(g, r + '.log'))}")


def k3bisect_section():
    print("## k3bisect (Kimi K3 debug, 1 GPU, deterministic, 10 steps, vs attn-gym v0.0.19)")
    d = os.path.join(OUT, "k3bisect")
    if not os.path.isdir(d):
        print("not run")
        return
    base_sha = "0d0197fb36d95695dccef03436bb0649f985b32d"
    commits = [base_sha] + read(os.path.join(d, "commits.txt")).split()
    ref = tb_scalars(os.path.join(d, "dump_" + base_sha, "tb_baseline"))
    first_diff = None
    for sha in commits:
        code = read(os.path.join(d, sha + ".exit"), "?").strip()
        subj = subprocess.run(
            ["git", "-C", AG_SRC, "log", "-1", "--format=%s", sha],
            capture_output=True, text=True,
        ).stdout.strip()[:60]
        a = tb_scalars(os.path.join(d, "dump_" + sha, "tb_baseline"))
        b = tb_scalars(os.path.join(d, "dump_" + sha, "tb_test"))
        if a is None or b is None or ref is None:
            status = f"CRASH exit={code} {last_error(os.path.join(d, sha + '.log'))[:80]}"
        else:
            repeat = a == b
            same = a == ref
            if not same and first_diff is None:
                first_diff = sha
            dl = max(abs(a["loss"][s] - ref["loss"][s]) for s in ref["loss"] if s in a["loss"])
            status = f"{'SAME' if same else 'DIFF'} repeat={'yes' if repeat else 'NO'} max|dloss|={dl:.1e}"
        print(f"{sha[:9]} {status} | {subj}")
    print(f"first commit that differs from v0.0.19: {first_diff[:9] if first_diff else 'none'}")


if __name__ == "__main__":
    print(f"# k31 h20 summary  ({OUT})")
    for section in (env_section, unit_section, ci_section, par_section, neg_section, k3bisect_section):
        try:
            section()
        except Exception as e:
            print(f"{section.__name__} failed: {type(e).__name__}: {e}")
