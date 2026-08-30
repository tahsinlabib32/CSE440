import csv, sys, json
csv.field_size_limit(sys.maxsize)
samples = {}
with open('legal_text_classification.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if row and len(row) > 3 and row[1] not in samples:
            samples[row[1]] = {'title': row[2], 'text': row[3]}
with open('samples.json', 'w', encoding='utf-8') as out:
    json.dump(samples, out)
