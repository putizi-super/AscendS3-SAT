# 初始化张量
n = 4
matrix = torch.zeros(n, n)

# 对角线索引
indices = torch.arange(n).unsqueeze(0)

# 对角线上的值
values = torch.tensor([1, 2, 3, 4])

# 在对角线插入值
matrix.scatter_(dim=1, index=indices, src=values.unsqueeze(1))
print(matrix)
