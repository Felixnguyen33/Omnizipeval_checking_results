import os, sys, json
import argparse

parser = argparse.ArgumentParser(description="LongBench v2 accuracy (%) per prediction file: Overall, Easy, Hard, Short, Medium, Long.")
parser.add_argument("--results_dir", default="results", help="Directory of pred.py *.jsonl files (default: results)")
parser.add_argument("--output", default="result.txt", help="Tab-separated table to write (default: result.txt)")
parser.add_argument("--expected", type=int, default=503, help="Warn when a file does not hold this many questions (default: 503)")
args = parser.parse_args()

files = sorted(os.listdir(args.results_dir))
output = ["Model\tOverall\tEasy\tHard\tShort\tMedium\tLong"]
compensated = False

for file in files:
    filename = os.path.join(args.results_dir, file)
    try:
        pred_data = json.load(open(filename, encoding='utf-8'))
        if isinstance(pred_data, dict):
            # A .jsonl file with exactly one line parses as a single JSON
            # object here rather than a list, so wrap it back into one.
            pred_data = [pred_data]
    except Exception as e:
        pred_data = [json.loads(line) for line in open(filename, encoding='utf-8') if line.strip()]

    if not pred_data:
        print(f"Skipping {file}: no records")
        continue
    if len(pred_data) != args.expected:
        # pred.py drops a question whose request failed; Overall is then over fewer questions.
        print(f"WARNING: {file} has {len(pred_data)} records, expected {args.expected}; run pred.py again to resume", file=sys.stderr)

    easy, hard, short, medium, long = 0, 0, 0, 0, 0
    easy_acc, hard_acc, short_acc, medium_acc, long_acc = 0, 0, 0, 0, 0
    for pred in pred_data:
        acc = int(pred['judge'])
        if compensated and pred["pred"] == None:
            acc = 0.25
        if pred["difficulty"] == "easy":
            easy += 1
            easy_acc += acc
        else:
            hard += 1
            hard_acc += acc

        if pred['length'] == "short":
            short += 1
            short_acc += acc
        elif pred['length'] == "medium":
            medium += 1
            medium_acc += acc
        else:
            long += 1
            long_acc += acc

    def pct(num, denom):
        return round(100 * num / denom, 1) if denom else float('nan')

    name = '.'.join(file.split('.')[:-1])
    output.append(name+'\t'+str(pct(easy_acc+hard_acc, len(pred_data)))+'\t'+str(pct(easy_acc, easy))+'\t'+str(pct(hard_acc, hard))+'\t'+str(pct(short_acc, short))+'\t'+str(pct(medium_acc, medium))+'\t'+str(pct(long_acc, long)))

open(args.output, 'w', encoding='utf-8').write('\n'.join(output))
print('\n'.join(output))
