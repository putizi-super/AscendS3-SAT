import numpy as np

def read_input_files():
    # Define file paths
    input_x_path = "./input/input_x.bin"
    input_y_path = "./input/input_y.bin"
    golden_path = "./output/golden.bin"

    # Read input_x.bin
    input_x = np.fromfile(input_x_path, dtype=np.float32)
    print("Contents of input_x.bin:")
    print(input_x)
    print("\nShape:", input_x.shape)
    
    # Read input_y.bin
    input_y = np.fromfile(input_y_path, dtype=np.float32)
    print("\nContents of input_y.bin:")
    print(input_y)
    print("\nShape:", input_y.shape)

    # Read golden.bin (contains boolean results of NotEqual operation)
    golden = np.fromfile(golden_path, dtype=np.bool_)
    print("\nContents of golden.bin (NotEqual results):")
    print(golden)
    print("\nShape:", golden.shape)

    # Print a detailed comparison
    # print("\nDetailed comparison:")
    # for i in range(len(input_x)):
    #     print(f"Index {i}:")
    #     print(f"  X: {input_x[i]}")
    #     print(f"  Y: {input_y[i]}")
    #     print(f"  NotEqual: {golden[i]}")

if __name__ == "__main__":
    read_input_files()
