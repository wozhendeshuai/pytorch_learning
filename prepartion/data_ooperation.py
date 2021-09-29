import torch
x=torch.arange(12)
print(x)
print(x.shape)
print(x.numel())
X=x.reshape(2,6)
print(X)
torch.zeros((2,3,4))
x=torch.tensor([1.0,2,4,8])
y=torch.tensor([2,2,2,2])
#x**y求幂
print(x + y, x - y, x * y, x ** y)