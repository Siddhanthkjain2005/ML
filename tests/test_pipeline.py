import csv
from pathlib import Path
from ber.pipeline import sample_queries, read_truth


def test_sample_is_order_independent_and_truth_preserves_multiple_matches(tmp_path):
    rows = [(f'S1-{i}', f'Business {i}', f'{i} Road', 'India') for i in range(1, 21)]
    paths = [tmp_path / name for name in ('forward.tsv', 'reverse.tsv')]
    for path, values in zip(paths, (rows, rows[::-1])):
        with path.open('w', newline='') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(['entity_id', 'business_name', 'business_address', 'country'])
            writer.writerows(values)
    first = sample_queries(paths[0], 7)
    assert first == sample_queries(paths[1], 7)
    truth_path = tmp_path / 'ground_truth.tsv'
    with truth_path.open('w', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['source1_entity_id', 'matched_entity_ids'])
        writer.writerows((row[0], 'S2-42,S3-43' if i else '') for i, row in enumerate(rows))
    full = sample_queries(paths[0], 0)
    truth = read_truth(truth_path, full)
    assert truth['S1-1'] == set()
    assert truth['S1-2'] == {'S2-42', 'S3-43'}
    assert len(truth) == 20
