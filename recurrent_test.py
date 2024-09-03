from typing import Literal
import numpy as np
import onnx

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_rnn_lstm_gru(
    op_type: Literal['RNN', 'LSTM', 'GRU'],
    input_size: int,
    hidden_size: int,
    seq_length: int,
    batch_size: int,
    num_directions: int = 1,
    **kwargs,
) -> None:
    num_layers = 1  # ONNX test assumes single-layer RNN/LSTM/GRU

    # Create input data
    x_shape = (seq_length, batch_size, input_size)
    x = np.random.uniform(low=-1.0, high=1.0, size=x_shape).astype(np.float32)

    # Create weights and biases
    W_shape = (num_directions, 4 * hidden_size if op_type == 'LSTM' else 1 * hidden_size, input_size)
    R_shape = (num_directions, 4 * hidden_size if op_type == 'LSTM' else 1 * hidden_size, hidden_size)
    B_shape = (num_directions, 8 * hidden_size if op_type == 'LSTM' else 2 * hidden_size)

    W = np.random.uniform(low=-1.0, high=1.0, size=W_shape).astype(np.float32)
    R = np.random.uniform(low=-1.0, high=1.0, size=R_shape).astype(np.float32)
    B = np.random.uniform(low=-1.0, high=1.0, size=B_shape).astype(np.float32)

    test_inputs = {'x': x}
    initializers = {'W': W, 'R': R, 'B': B}

    # Create the ONNX node
    inputs = ['x', 'W', 'R', 'B']
    if op_type == 'LSTM':
        H0_shape = (num_layers * num_directions, batch_size, hidden_size)
        C0_shape = (num_layers * num_directions, batch_size, hidden_size)
        H0 = np.random.uniform(low=-1.0, high=1.0, size=H0_shape).astype(np.int32)
        C0 = np.random.uniform(low=-1.0, high=1.0, size=C0_shape).astype(np.float32)
        initializers.update({'H0': H0, 'C0': C0})
        inputs.extend(['H0', 'C0'])

    elif op_type in ['RNN', 'GRU']:
        H0_shape = (num_layers * num_directions, batch_size, hidden_size)
        H0 = np.random.uniform(low=-1.0, high=1.0, size=H0_shape).astype(np.int32)
        initializers.update({'H0': H0})
        inputs.append('H0')

    outputs = ['y', 'y_h'] if op_type != 'LSTM' else ['y', 'y_h', 'y_c']

    node = onnx.helper.make_node(
        op_type=op_type,
        inputs=inputs,
        outputs=outputs,
        hidden_size=hidden_size,
        **kwargs,
    )

    print(f'inputs = {inputs}')
    print(f'outputs = {outputs}')

    for k,v in initializers.items():
        print(f'k={k},v={v.shape}')

    # Create the ONNX model
    model = make_model_from_nodes(nodes=node, initializers=initializers, inputs_example=test_inputs)
    
    # STOP

    # Validate the model
    check_onnx_model(
        model,
        test_inputs,
        atol_onnx_torch=10**-4,
        atol_torch_cpu_cuda=10**-4,
    )


# Example usage to test RNN, LSTM, and GRU support
_test_rnn_lstm_gru(
    op_type='RNN',
    input_size=10,
    hidden_size=20,
    seq_length=5,
    batch_size=3,
    num_directions=1,
)
"""
_test_rnn_lstm_gru(
    op_type='LSTM',
    input_size=10,
    hidden_size=20,
    seq_length=5,
    batch_size=3,
    num_directions=1,
)

_test_rnn_lstm_gru(
    op_type='GRU',
    input_size=10,
    hidden_size=20,
    seq_length=5,
    batch_size=3,
    num_directions=1,
)
"""