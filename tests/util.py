import numpy as np

def create_data(shape, dtype):
    if np.issubdtype(dtype, np.floating):
        data = np.random.rand(*shape) + 1.0
        return data.astype(dtype)
    else:
        data = np.random.randint(255, size=shape)
        return data.astype(dtype)

def print_close_broken_elements(torch_result, ov_tensor, tolerance=1e-02):
    # Perform element-wise comparison using np.isclose
    close_elements = np.isclose(torch_result, ov_tensor, rtol=tolerance)

    # Count the number of close elements
    count_close_elements = np.sum(close_elements)

    # Print the number of close elements
    print(f'Number of close elements: {count_close_elements}')

    # Calculate the total number of elements
    total_elements = np.prod(ov_tensor.shape)

    # Calculate the number of broken (not close) elements
    count_broken_elements = total_elements - count_close_elements

    # Print the number of broken elements
    print(f'Number of broken elements: {count_broken_elements}')

