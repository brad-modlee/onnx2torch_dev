__all__ = []

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxMapping
from onnx2torch.utils.common import OperationConverterResult

_RNN_CLASS_FROM_TYPE = {
    'RNN': nn.RNN,
    'LSTM': nn.LSTM,
    'GRU': nn.GRU,
}

@add_converter(operation_type='RNN', version=1)
@add_converter(operation_type='LSTM', version=1)
@add_converter(operation_type='GRU', version=1)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:

    print('called converter RNN')
    STOP

    # Extract weights and biases from the graph
    W_value_name = node.input_values[1]
    R_value_name = node.input_values[2]
    B_value_name = node.input_values[3] if len(node.input_values) > 3 else None
    H0_value_name = node.input_values[4] if len(node.input_values) > 4 else None
    C0_value_name = node.input_values[5] if node.operation_type == 'LSTM' and len(node.input_values) > 5 else None

    if W_value_name not in graph.initializers or R_value_name not in graph.initializers:
        raise Exception(f"Graph does not have necessary weight tensors: {W_value_name} or {R_value_name}")

    W = graph.initializers[W_value_name].to_torch()
    R = graph.initializers[R_value_name].to_torch()

    if B_value_name is not None and B_value_name in graph.initializers:
        B = graph.initializers[B_value_name].to_torch()
    else:
        B = None

    if H0_value_name is not None and H0_value_name in graph.initializers:
        H0 = graph.initializers[H0_value_name].to_torch()
    else:
        H0 = None

    if C0_value_name is not None and C0_value_name in graph.initializers:
        C0 = graph.initializers[C0_value_name].to_torch()
    else:
        C0 = None

    # Debugging: Print shapes and types to verify they are correct
    print(f"RNN Type: {node.operation_type}")
    print(f"W Shape: {W.shape}")
    print(f"R Shape: {R.shape}")
    if B is not None:
        print(f"B Shape: {B.shape}")
    if H0 is not None:
        print(f"H0 Shape: {H0.shape}")
    if C0 is not None:
        print(f"C0 Shape: {C0.shape}")

    # Determine the number of layers, directions, and hidden size
    hidden_size = node.attributes['hidden_size']
    num_directions = W.shape[0]
    num_layers = 1  # ONNX default for single-layer RNN/LSTM/GRU

    # Create the appropriate RNN module
    rnn_type = node.operation_type
    rnn_class = _RNN_CLASS_FROM_TYPE[rnn_type]

    torch_module = rnn_class(
        input_size=W.shape[2], 
        hidden_size=hidden_size, 
        num_layers=num_layers, 
        bias=B is not None,
        batch_first=False,  # ONNX uses sequence-first format
        bidirectional=(num_directions == 2)
    )

    with torch.no_grad():
        # Set weights and biases
        torch_module.weight_ih_l0 = W.view_as(torch_module.weight_ih_l0)
        torch_module.weight_hh_l0 = R.view_as(torch_module.weight_hh_l0)
        if B is not None:
            torch_module.bias_ih_l0 = B[:hidden_size * num_directions].view_as(torch_module.bias_ih_l0)
            torch_module.bias_hh_l0 = B[hidden_size * num_directions:].view_as(torch_module.bias_hh_l0)

    # If H0 and C0 are part of the model, they need to be handled in the forward pass as well
    # For this specific converter, we just handle their shapes and presence here
    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=OnnxMapping(
            inputs=(node.input_values[0],),
            outputs=node.output_values,
        ),
    )
