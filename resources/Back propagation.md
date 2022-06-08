# Back propagation

## Letters:

- L = Total number of layers in the network
- $S_l$=Number of units(not counting bias unit) in layer l
- K = Classes/Number of possible predictions/Number of units in the last layer

## Not Regularized Cost function:

$$
J(\theta) = \frac1m \sum_{i=1}^m \sum_{k=1}^K[-y_k^{(i)}log((h_\theta(x^{(i)}))_k) - (1 - y_k^{(i)})log(1 - (h_\theta(x^{(i)}))_k)]
$$

## Regularized Cost function:

$$
J(\theta) = -\frac{1}{m}[\sum_{i=1}^{m} \sum_{k=1}^{K}y_k^{(i)}log(h_\theta(x^{(i)}))_k + (1 - y_k^{(i)})log(1-(h_\theta(x^{(i)}))_k)] + \frac{\lambda}{2m}\sum_{l=1}^{L - 1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_l + 1}(\theta_{ji}^{(l)})^2   
$$

Basically what it does is: it sums every class’ costs for every training example. so for a given training example it sums the costs of every class.

$$
g'(z) = g(z)(1 - g(z))
$$

$$
\delta^{(L)} = a^{(L)} - y^{(i)}
$$

$$
\delta^{(l)} = (\theta^{(l)})^T\delta^{(l+1)} .* g'(z^{(l)})
$$

(Remove $\delta^{(l)}_0$)

$$
\Delta^{(l)} := \Delta^{(l)} + \delta^{(l+1)}(a^{(l)})^T
$$

The new cost of a layer is equal to the old cost of the same layer(initiates at 0) plus the cost of the next layer times the values of this layer.

Example: The cost of layer A is equal to the cost for the previous example + the cost of layer B times the values of layer A

## How to implement it:

Training set {($x^{(1)}$, $y^{(1)}$), …, ($x^{(m)}, y^{(m)}$)}

Set $\Delta_{ij}^{(l)} = 0$ (for all l, i, j).

For i=1 to m

Set $a^{(1)} = x^{(i)}$

Perform forward propagation to compute $a^{(l)}$ for $l = 2, 3, ..., L$ 

Using $y^{(i)}$, compute $\delta^{(L)} = a^{(L)} - y^{(i)}$

Compute $\delta^{(L-1)}, \delta^{(L-2)}... \delta^{(2)}$

$\Delta^{(l)} := \Delta^{(l)} + \delta^{(l + 1)}(a^{(l)})$

$$
J'(\theta) = D = \begin{cases}
{\frac1m}\Delta_{ij}^{(l)} +\frac\lambda{m}\theta_{ij}^{(l)} &\text{if } j \ge 1\\
\frac1m \Delta_{ij}^{(l)} &\text{if } j = 0 \\
\end{cases}
$$

## Gradient Checking:

This is a way to make sure that there is no bug in the backpropagation’s implementation.

A good value for EPSILON = $10^{-4}$

for i = 1:n

thetaPlus = theta;

thetaPlus(i) = thetaPlus(i) + EPSILON;

thetaMinus = theta;

thetaMinus(i) = thetaMinus - EPSILON;

gradApprox(i) = (J(thetaPlus) - J(thetaMinus)) / (2 * EPSILON);

end;

Then check if gradApprox $\approx$ DVec

Where DVec is from back propagation

## Implementation Note:

- Implement back propagation to compute ***DVec**(unrolled $D^{(1)}$, $D^{(2)}$,$D^{(3)}$).*
- Implement numerical gradient check to compute **gradApprox.**
- Make sure they give similar values.
- **Turn off gradient checking**. Using backprop code for learning.

## Important:

- **Be sure to disable your gradient checking code before training your classifier**. If you run numerical gradient computation on every iteration of gradient descent your code will be very slow

## How to initialize theta:

It is VERY important to initialize all theta randomly.