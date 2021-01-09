# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import torch


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    # Directly
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    print(x_data)

    # from Numpy
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
    print(x_np)

    # From another tensor:
    x_ones = torch.ones_like(x_data)
    print(f"Ones Tensor:\n{x_ones}\n")
    x_rand = torch.rand_like(x_data, dtype=torch.float)
    print(f"Random Tensor:\n{x_rand}\n")

    # With random or constant values:
    shape = (2, 3,)
    rand_tensor = torch.rand(shape)
    print(f"Random Tensor:\n{rand_tensor}")
    ones_tensor = torch.ones(shape)
    print(f"Ones Tensor:\n{ones_tensor}")
    zeros_tensor = torch.zeros(shape)
    print(f"Zeros Tensor:\n{zeros_tensor}")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
