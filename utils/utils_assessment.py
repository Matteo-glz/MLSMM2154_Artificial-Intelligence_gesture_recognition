from collections import Counter


def majority_vote(neighbors):
    labels = [n[1] for n in neighbors]
    return Counter(labels).most_common(1)[0][0]