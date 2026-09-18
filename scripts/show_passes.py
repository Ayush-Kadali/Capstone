"""Inspect retained compiler artifacts. Pure stdlib; reads report without changing it."""
import argparse
import difflib
import json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("report")
parser.add_argument("--policy", default="cost")
parser.add_argument("--pass-name", help="Only show this pass")
parser.add_argument("--full", action="store_true", help="Print full before and after IR instead of a diff")
args = parser.parse_args()
with open(args.report) as handle:
    report = json.load(handle)
results = [r for r in report["results"] if r["policy"] == args.policy]
if not results:
    parser.error("Policy has no feasible recorded result")
passes = report["frontend_passes"] + results[0]["passes"]
if args.pass_name:
    passes = [p for p in passes if p["name"] == args.pass_name]
    if not passes:
        parser.error("Pass name not found")
for record in passes:
    print(f"\n=== {record['name']} ===")
    for field in ("changes", "reasons", "consequences"):
        print(field.upper())
        for item in record[field]:
            print("  " + item)
    if args.full:
        print("BEFORE\n" + record["input_ir"])
        print("AFTER\n" + record["output_ir"])
    else:
        print("".join(difflib.unified_diff(record["input_ir"].splitlines(True), record["output_ir"].splitlines(True), fromfile="before", tofile="after")), end="")
