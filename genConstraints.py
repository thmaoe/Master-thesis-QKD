from itertools import product
from collections import defaultdict

# Operator list (length 28)
S = [
    "id", "M0", "M1", "M2",
    "Z00", "Z00dag", "Z10", "Z10dag", "Z20", "Z20dag",
    "Z01", "Z01dag", "Z11", "Z11dag", "Z21", "Z21dag",
    "M0Z00", "M0Z00dag", "M0Z01", "M0Z01dag",
    "M1Z10", "M1Z10dag", "M1Z11", "M1Z11dag",
    "M2Z20", "M2Z20dag", "M2Z21", "M2Z21dag"
]

def dagger(op):
    if op == "id": return "id"
    if "dag" in op: return op.replace("dag", "")
    elif op.startswith("Z"): return op + "dag"
    elif op.startswith("M"): return op
    elif "M" in op and "Z" in op:
        m, z = op[:2], op[2:]
        return m + dagger(z)
    return op

def split_op(op):
    if op == "id": return ["id"]
    if "Z" in op and "M" in op:
        return [op[:2], op[2:]]
    return [op]

# Simplify using [Mb, Zax] = 0
def simplify_commutation(seq):
    reordered = seq[:]
    changed = True
    while changed:
        changed = False
        for i in range(len(reordered) - 1):
            a, b = reordered[i], reordered[i + 1]
            if a.startswith("Z") and b.startswith("M"):
                reordered[i], reordered[i + 1] = b, a
                changed = True
    return tuple(reordered)

S_dagger = [split_op(dagger(op)) for op in S]
S_raw = [split_op(op) for op in S]

# Map to simplified operator sequences
product_map = defaultdict(list)
for k, l in product(range(len(S)), repeat=2):
    composed = S_dagger[k] + S_raw[l]
    simplified = simplify_commutation(composed)
    product_map[simplified].append((k, l))

# Equality constraints
equality_constraints = set()
for indices in product_map.values():
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            a, b = indices[i]
            c, d = indices[j]
            equality_constraints.add(((a, b), (c, d)))

# Zero constraints from Mb Mb' = 0 for b ≠ b'
def extract_M(oplist):
    for op in oplist:
        if op.startswith("M"):
            return op
    return None

zero_constraints = set()
for k, l in product(range(len(S)), repeat=2):
    mk = extract_M(S_raw[k])
    ml = extract_M(S_raw[l])
    if mk and ml and mk != ml:
        zero_constraints.add((k, l))

# Function to emit block-extended constraints
def emit_block_constraints(base_indices, expr):
    shifts = [(0, 0), (28, 0), (0, 28), (28, 28)]
    for dx1, dy1 in shifts:
        for dx2, dy2 in shifts:
            i1, j1 = base_indices[0][0] + dx1, base_indices[0][1] + dy1
            i2, j2 = base_indices[1][0] + dx2, base_indices[1][1] + dy2
            print(f"constraints += [Gamma[{i1}, {j1}] == Gamma[{i2}, {j2}]]")

# Output zero constraints
print("# Zero constraints (with blocks)")
for k, l in sorted(zero_constraints):
    for dx, dy in [(0, 0), (28, 0), (0, 28), (28, 28)]:
        i, j = k + dx, l + dy
        print(f"constraints += [Gamma[{i}, {j}] == 0]")

# Output equality constraints
print("\n# Equality constraints (with blocks)")
for (k1, l1), (k2, l2) in sorted(equality_constraints):
    emit_block_constraints(((k1, l1), (k2, l2)), "==")
