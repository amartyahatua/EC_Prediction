from collections import Counter
from datasets import load_dataset

dataset = load_dataset("lightonai/SwissProt-EC-leaf", split="train")

ec_counts = Counter()
for temp in dataset:
    if len(temp['seq']) <= 512:
        ec = temp['labels_str'][0] if isinstance(temp['labels_str'], list) else temp['labels_str']
        ec_counts[ec] += 1

print("Top 20 EC classes by count:")
for ec, count in ec_counts.most_common(20):
    print(f"  {ec}: {count} proteins")


# ['EC:2.7.7.6']: 1368 proteins
# ['EC:7.1.2.2']: 1208 proteins
# ['EC:3.6.4.12']: 1144 proteins
# ['EC:5.2.1.8']: 1142 proteins
# ['EC:3.1.26.4']: 913 proteins
# ['EC:4.2.1.33']: 838 proteins
# ['EC:7.1.1.2']: 825 proteins
# ['EC:4.2.1.20']: 813 proteins
# ['EC:2.1.3.15']: 779 proteins
# ['EC:2.7.11.1']: 758 proteins
# ['EC:4.2.1.11']: 732 proteins
# ['EC:2.1.1.199']: 687 proteins
# ['EC:2.7.4.3']: 675 proteins
# ['EC:1.5.1.5', 'EC:3.5.4.9']: 668 proteins
# ['EC:3.1.11.6']: 664 proteins
# ['EC:6.3.4.4']: 650 proteins
# ['EC:6.1.1.11']: 639 proteins
# ['EC:2.5.1.75']: 634 proteins
# ['EC:6.1.1.17']: 622 proteins
# ['EC:2.1.2.1']: 616 proteins