#include <iostream>
#include <vector>
#include <stdexcept>
#include <iomanip>

// 模板类Tensor，支持不同数据类型
template<typename T>
class Tensor {
public:
    std::vector<T> data;          // 存储张量数据的一维数组
    std::vector<size_t> shape;    // 存储张量每个维度的大小
    std::vector<size_t> strides;  // 存储每个维度的步长

    // 构造函数：初始化张量的形状和数据
    Tensor(const std::vector<size_t>& shape_, const std::vector<T>& data_) 
        : shape(shape_), data(data_) {
        computeStrides();
    }

    // 计算张量中每个维度的步长
    // 步长表示在特定维度上移动一个单位需要跨越的元素个数
    void computeStrides() {
        strides.resize(shape.size());
        size_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

    // 根据多维索引获取张量中的值
    T get_value(const std::vector<size_t>& indices) const {
        size_t offset = 0;
        // 将多维索引转换为一维数组的偏移量
        for (size_t i = 0; i < shape.size(); i++) {
            offset += indices[i] * strides[i];
        }
        return data[offset];
    }

    // 打印张量内容（仅支持3维张量的打印）
    void print() const {
        if (shape.size() == 3) {
            size_t dim0 = shape[0], dim1 = shape[1], dim2 = shape[2];
            for (size_t i = 0; i < dim0; ++i) {
                std::cout << "[\n";
                for (size_t j = 0; j < dim1; ++j) {
                    std::cout << " [";
                    for (size_t k = 0; k < dim2; ++k) {
                        std::cout << std::setw(3) << data[i * strides[0] + j * strides[1] + k * strides[2]];
                        if (k < dim2 - 1) std::cout << ", ";
                    }
                    std::cout << "]\n";
                }
                std::cout << "]\n";
            }
        }
    }

    // scatter_操作的主函数
    void scatter_(int dim, const Tensor<size_t>& index, const Tensor<T>& src) {
        // 检查维度的有效性
        if (dim < 0 || dim >= shape.size()) {
            throw std::runtime_error("Invalid dimension");
        }
        
        // 检查所有张量的维度数量是否相同
        if (index.shape.size() != shape.size() || src.shape.size() != shape.size()) {
            throw std::runtime_error("All tensors must have the same number of dimensions");
        }
        
        // 检查维度大小约束
        for (size_t d = 0; d < shape.size(); d++) {
            if (index.shape[d] > src.shape[d]) {
                throw std::runtime_error("index.size(d) must be <= src.size(d) for all dimensions");
            }
            if (d != dim && index.shape[d] > shape[d]) {
                throw std::runtime_error("index.size(d) must be <= self.size(d) for all dimensions d != dim");
            }
        }
        
        // 开始递归处理scatter操作
        std::vector<size_t> current_indices(shape.size(), 0);
        scatter_recursive(dim, index, src, current_indices, 0);
    }

private:
    // scatter操作的递归实现
    void scatter_recursive(int dim, const Tensor<size_t>& index, const Tensor<T>& src,
                         std::vector<size_t>& current_indices,
                         int current_dim) {
        // 当遍历到最后一个维度时，执行实际的值拷贝
        if (current_dim == shape.size()) {
            size_t src_offset = 0;  // 源张量的偏移量
            size_t dst_offset = 0;  // 目标张量的偏移量
            
            // 计算源和目标的偏移量
            for (size_t i = 0; i < shape.size(); i++) {
                if (i == dim) {
                    // 在指定维度使用index中的值作为索引
                    size_t idx = index.get_value(current_indices);
                    if (idx >= shape[i]) {
                        throw std::runtime_error("Index out of bounds");
                    }
                    dst_offset += idx * strides[i];
                } else {
                    dst_offset += current_indices[i] * strides[i];
                }
                src_offset += current_indices[i] * src.strides[i];
            }
            
            // 执行值拷贝
            data[dst_offset] = src.data[src_offset];
            return;
        }
        
        // 递归遍历所有维度
        for (size_t i = 0; i < index.shape[current_dim]; i++) {
            current_indices[current_dim] = i;
            scatter_recursive(dim, index, src, current_indices, current_dim + 1);
        }
    }
};

int main() {
    // 创建一个2x3x3的目标张量，初始值全为0
    std::vector<size_t> self_shape = {2, 3, 3};
    std::vector<float> self_data(18, 0);  // 2*3*3 = 18个元素
    Tensor<float> self(self_shape, self_data);

    // 创建一个2x2x2的源张量，包含值1-8
    std::vector<size_t> src_shape = {2, 2, 2};
    std::vector<float> src_data = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor<float> src(src_shape, src_data);

    // 创建索引张量，指定要写入的位置
    std::vector<size_t> index_shape = {2, 2, 2};
    std::vector<size_t> index_data = {0, 1, 1, 2, 0, 1, 1, 2};
    Tensor<size_t> index(index_shape, index_data);

    // 打印原始张量
    std::cout << "Original tensor:\n";
    self.print();

    // 在维度1上执行scatter操作
    self.scatter_(1, index, src);

    // 打印scatter后的结果
    std::cout << "\nAfter scatter:\n";
    self.print();

    return 0;
}