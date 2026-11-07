def distance(a, b):
    return (int(a, 16) ^ int(b, 16)).bit_count()


def group_rows(rows, max_distance=6):
    """Union near-identical perceptual hashes. rows: id, dhash, size, taken_at, path, thumb, favorite."""
    items = [r for r in rows if r["dhash"]]
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(items)):
        hi = int(items[i]["dhash"], 16)
        for j in range(i + 1, len(items)):
            if (hi ^ int(items[j]["dhash"], 16)).bit_count() <= max_distance:
                union(i, j)

    buckets = {}
    for i, item in enumerate(items):
        buckets.setdefault(find(i), []).append(item)

    groups = [sorted(g, key=lambda r: r["id"]) for g in buckets.values() if len(g) > 1]
    groups.sort(key=lambda g: (-len(g), g[0]["id"]))
    return groups
