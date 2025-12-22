import onnx

def fix_and_cleanup_unet(model_path, output_path):
    model = onnx.load(model_path)
    graph = model.graph
    
    # 1. Define our target input
    new_time_emb_name = '/down_blocks.0/resnets.0/act_1/Mul_output_0'
    
    # 2. Find the 1280-dim embedding output. 
    # In SD1.5, this is usually the output of the second SiLU in the time embedding.
    # We found it was 'silu_2' in our previous inspection.
    old_time_emb_name = 'silu_2'
    
    # 3. Replace all usages of 'silu_2' with the new name
    count = 0
    for node in graph.node:
        for i, input_name in enumerate(node.input):
            if input_name == old_time_emb_name:
                node.input[i] = new_time_emb_name
                count += 1
    print(f"Replaced {count} usages of {old_time_emb_name} with {new_time_emb_name}")
    
    # 4. Add the new name as a graph input
    # We'll use a fixed shape [1, 1280] as it's standard for SD1.5
    new_input = onnx.helper.make_tensor_value_info(
        new_time_emb_name,
        onnx.TensorProto.FLOAT,
        [1, 1280]
    )
    
    # Remove existing input with same name if any
    graph.input.extend([new_input])
    
    # 5. Remove 't' from graph inputs
    t_inputs = [i for i in graph.input if i.name == 't']
    for i in t_inputs:
        graph.input.remove(i)
    
    # 6. Recursively remove all nodes that have missing inputs
    while True:
        inputs = {inp.name for inp in graph.input}
        initializers = {init.name for init in graph.initializer}
        node_outputs = {out for node in graph.node for out in node.output}
        all_available = inputs | initializers | node_outputs
        
        to_remove = []
        for node in graph.node:
            if any(inp and inp not in all_available for inp in node.input):
                to_remove.append(node)
        
        if not to_remove:
            break
            
        print(f"Removing {len(to_remove)} dangling nodes...")
        for node in to_remove:
            graph.node.remove(node)
            
    # 7. Remove duplicate inputs
    seen_inputs = set()
    unique_inputs = []
    for inp in graph.input:
        if inp.name not in seen_inputs:
            unique_inputs.append(inp)
            seen_inputs.add(inp.name)
    
    while len(graph.input) > 0:
        graph.input.pop()
    graph.input.extend(unique_inputs)

    onnx.save(model, output_path, save_as_external_data=True)
    print(f"Saved fixed and cleaned model to {output_path}")

if __name__ == "__main__":
    # We need to start from the RAW exported model if possible, 
    # but since we already modified it, let's hope it still has the nodes we need to redirect.
    # Actually, if silu_2 is gone, we might need to find another way.
    fix_and_cleanup_unet("onnx-models/unet.onnx", "onnx-models/unet_fixed.onnx")
