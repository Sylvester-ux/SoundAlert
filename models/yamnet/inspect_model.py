import tflite

MODEL_PATH = "yamnet.tflite"

with open(MODEL_PATH, "rb") as f:
    model_data = f.read()

model = tflite.Model.GetRootAsModel(model_data, 0)

print("=" * 60)
print("SOUNDALERT — YAMNET TFLITE MODEL INSPECTION")
print("=" * 60)
print("Schema version:", model.Version())
print("Subgraphs:", model.SubgraphsLength())
print("Operator codes:", model.OperatorCodesLength())

subgraph = model.Subgraphs(0)

def tensor_type_name(value):
    try:
        return tflite.TensorType.Name(value)
    except Exception:
        return str(value)

def inspect_tensor(label, tensor_index):
    tensor = subgraph.Tensors(tensor_index)
    shape = [
        tensor.Shape(i)
        for i in range(tensor.ShapeLength())
    ]

    print(f"\n{label}")
    print("  Tensor index:", tensor_index)
    print("  Name:", tensor.Name().decode("utf-8", errors="replace"))
    print("  Shape:", shape)
    print("  Type:", tensor_type_name(tensor.Type()))

    quant = tensor.Quantization()
    if quant:
        scales = [
            quant.Scale(i)
            for i in range(quant.ScaleLength())
        ]
        zero_points = [
            quant.ZeroPoint(i)
            for i in range(quant.ZeroPointLength())
        ]
        print("  Quantization scales:", scales)
        print("  Quantization zero points:", zero_points)
        print("  Quantized dimension:", quant.QuantizedDimension())
    else:
        print("  Quantization: none")

print("\nINPUTS")
for i in range(subgraph.InputsLength()):
    inspect_tensor("Input " + str(i), subgraph.Inputs(i))

print("\nOUTPUTS")
for i in range(subgraph.OutputsLength()):
    inspect_tensor("Output " + str(i), subgraph.Outputs(i))

print("\nOPERATOR LIST")
seen = set()

for i in range(model.OperatorCodesLength()):
    code = model.OperatorCodes(i)
    builtin = code.BuiltinCode()
    custom = code.CustomCode()

    if custom:
        name = custom.decode("utf-8", errors="replace")
    else:
        try:
            name = tflite.BuiltinOperator.Name(builtin)
        except Exception:
            name = f"BUILTIN_{builtin}"

    seen.add(name)

for name in sorted(seen):
    print(" ", name)

print("\nOperator count:", subgraph.OperatorsLength())
print("Unique operator types:", len(seen))
print("\nInspection complete.")
