y(out) = ax + b

a = 0.9141 (linear.weight)

b = -0.9166 (linear.bias)

loss = 9890.837891??

train ?? 

x        =   1, 2, 3, 4, 100, 200

y target =   3, 5, 7, 9, 201, 401

y (out） =   1.389	2.3031	3.2172	4.1313	91.8849	183.2949
target-out= -1.611	-2.6969	-3.7828	-3.8687	-109.1151	-217.7051
^2        =  2.595321	7.27326961	14.30957584	14.96683969	11906.10505	47395.51057
sum       =   59340.76062
sum / n   =   9890.12677 = loss




Epoch[3/3], loss: 9890.837891
('linear.weight', Parameter containing:
tensor([[0.9141]], requires_grad=True))
('linear.bias', Parameter containing:
tensor([0.4749], requires_grad=True))
tensor([[  1.3890],
        [  2.3032],
        [  3.2173],
        [  4.1314],
        [ 91.8890],
        [183.3031]], grad_fn=<AddmmBackward0>)