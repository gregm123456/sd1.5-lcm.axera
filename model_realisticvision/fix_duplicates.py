import onnx

def remove_duplicate_inputs(model_path, output_path):
    model = onnx.load(model_path)
    graph = model.graph
    
    seen_inputs = set()
    unique_inputs = []
    for inp in graph.input:
        if inp.name not in seen_inputs:
            unique_inputs.append(inp)
            seen_inputs.add(inp.name)
        else:
            print(f"Removing duplicate input: {inp.name}")
            
    # Clear and re-add
    while len(graph.input) > 0:
        graph.input.pop()
    graph.input.extend(unique_inputs)
    
    onnx.save(model, output_path, save_as_external_data=True)
    print(f"Saved fixed model to {output_path}")

if __name__ == "__main__":
    remove_duplicate_inputs("onnx-models/unet.onnx", "onnx-models/unet_fixed.onnx")
