import tflite
from collections import Counter

with open("yamnet.tflite", "rb") as f:
    data = f.read()

model = tflite.Model.GetRootAsModel(data, 0)
sg = model.Subgraphs(0)

print("OPERATOR CODE TABLE")
for i in range(model.OperatorCodesLength()):
    code = model.OperatorCodes(i)
    builtin = code.BuiltinCode()
    deprecated = code.DeprecatedBuiltinCode()
    version = code.Version()

    try:
        name = tflite.BuiltinOperator.Name(builtin)
    except Exception:
        name = f"BUILTIN_{builtin}"

    print(
        f"opcode_index={i} name={name} "
        f"builtin={builtin} deprecated={deprecated} "
        f"version={version}"
    )

print("\nOPERATOR USAGE")
counts = Counter()
for i in range(sg.OperatorsLength()):
    op = sg.Operators(i)
    opcode_index = op.OpcodeIndex()
    code = model.OperatorCodes(opcode_index)

    try:
        name = tflite.BuiltinOperator.Name(code.BuiltinCode())
    except Exception:
        name = f"BUILTIN_{code.BuiltinCode()}"

    counts[name] += 1
    print(f"node={i} opcode_index={opcode_index} name={name}")

print("\nCOUNTS")
for name, count in sorted(counts.items()):
    print(f"{name}: {count}")

print("\nOUTPUT TENSOR")
for i in range(sg.OutputsLength()):
    index = sg.Outputs(i)
    tensor = sg.Tensors(index)
    print("Output tensor index:", index)
    print("Output tensor name:", tensor.Name().decode(errors="replace"))
    print("Producer nodes are listed above; inspect the final graph operators.")
