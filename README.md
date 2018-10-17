
# The chain rule

### Introduction

So far we have seen that the derivative of a function is the instantaneous rate of change of that function.  In other words, how does a function's output change as we change one of the variables.  In this lesson, we will learn about the chain rule, which allows us to see how a function's output change as we change a variable that function does not directly depend on.  The chain rule may seem complicated, but it is just a matter of following the prescribed procedure.  Learning about the chain rule will allow us to take the derivative of more complicated functions that we will encounter in machine learning.

### The chain rule

Ok, now let's talk about the chain rule.  Imagine that we would like to take the derivative of the following function:

$$f(x) = (.5x + 3)^2 $$

Doing something like that can be pretty tricky right off the bat.  Lucky for us, we can use the chain rule.  The chain rule is essentially a trick that can be applied when our functions get complicated.  The first step is using functional composition to break our function down. Ok, let's do it.

$$g(x) = .5x + 3 $$
$$f(x) = (g(x))^2$$

Let's turn these two into functions while we are at it.


```python
def g_of_x(x):
    return .5*x + 3
```


```python
g_of_x(2) # 4
```




    4.0




```python
def f_of_x(x):
    return (g_of_x(x))**2
```


```python
f_of_x(2) # 16
```




    16.0



Looking at both the mathematical and code representations of $f(x)$ and $g(x)$, we can see that the $f(x)$ function wraps the $g(x)$ function.  So let's call $f(x)$ **the outer function**, and $g(x)$ **the inner function**.

```python

def g_of_x(x):
    return .5*x + 3

def f_of_x(x): # outer function f(x)
    return (g_of_x(x))**2 #inner function g(x)

```

Let's plot these two functions.


```python
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)

from graph import trace_values, plot

x_values = list(range(0, 10))
f_of_x_values = list(map(lambda x: f_of_x(x),x_values))
g_of_x_values = list(map(lambda x: g_of_x(x),x_values))
f_of_x_trace = trace_values(x_values, f_of_x_values, mode = 'lines', name = 'g(x) = .5*x')
g_of_x_trace = trace_values(x_values, g_of_x_values, mode = 'lines', name = 'f(x) = (g(x))^2')
plot([g_of_x_trace, f_of_x_trace])
```


<script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window.Plotly) {{require(['plotly'],function(plotly) {window.Plotly=plotly;});}}</script>



<div id="653c3861-6c12-4506-9942-03e5205d4d85" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("653c3861-6c12-4506-9942-03e5205d4d85", [{"x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "y": [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5], "mode": "line", "name": "f(x) = (g(x))^2", "text": []}, {"x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "y": [9.0, 12.25, 16.0, 20.25, 25.0, 30.25, 36.0, 42.25, 49.0, 56.25], "mode": "line", "name": "g(x) = .5*x", "text": []}], {}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>


Ok, so now that we have a sense of how our function $g(x) = .5x$ and $f(x) = (g(x))^2$ look, let's begin to take derivatives of these functions, starting with the derivative of $g(x)$, the inner function.

From our rules about derivatives we know that the power rule tells us that the derivative of $g(x) = .5x$ is

$$g'(x) = 1*.5x^0 = .5$$

Now a trickier question is what is the derivative of, our outer function $f(x)$?  So how does the output of our outer function, $f(x)$, change as we vary $x$.  

Notice that the outer function $f(x)$'s output does not directly vary with $x$.  Instead, it's output varies based on the output, $g(x)$, whose output varies with $x$.

```python

def g_of_x(x):
    return .5*x + 3

def f_of_x(x): # outer function f(x)
    return (g_of_x(x))**2 #inner function g(x)

```

> ** The chain rule**: So in taking the derivative, $\frac{\Delta f}{\Delta x}$ of an outer function, $f(x)$, which depends on an inner function $g(x)$, which depends on $x$, the derivative equals the derivative of the outer function times the derivative of the inner function.  

**Or: **

$$ f'(g(x)) = f'g(x)*g'(x) $$

### Work through our steps

Ok, so that is the chain rule.  Let's apply this to our example.

#### 1. Separate the function into two functions

Remember we started with the function $f(x) = (.5x + 3)^2 $.  Then we used functional composition to split this into two.

$$g(x) = x^2$$
$$f(x) = (g(x))^2$$

**2. Find the derivatives, $f'(x)$ and $g'(x)$**

* as we know $g'(x) = .5$
* and $f'g((x)) = 2*(g(x))^{1} = 2*g(x)$

** 3. Substitute into our chain rule **

We have:
* $ f'(g(x)) = f'g(x)*g'(x) = 2*g(x)*.5 = 1*g(x)$

Then substituting for $g(x)$, which we already defined, we have:

$f'(g(x)) = g(x) = x^2$

So the derivative of the function $f(x) = (.5x + 3)^2 $ is $f'(x) = x^2 $

### Say it again

The chain rule is allows us to the rate of change of a function that does not directly depend on a variable, $x$, but rather depends on a separate function, that depends on $x$.  For example, the function $f(x)$ below.

```python

def g_of_x(x):
    return .5*x + 3

def f_of_x(x): # outer function f(x)
    return (g_of_x(x))**2 #inner function g(x)

```

It does not directly depend on $x$, but depends on a function $g(x)$, which varies with different outputs of $x$.  So now we want to take the derivative of $f(x)$.

> Remember, taking a derivative means changing a variable $x$ a little, and seeing the change in the output.  The chain rule allows us to solve the problem of seeing the change in output when our function does not **directly** depend on that changing variable, but depends on **a function ** that depends on a variable.  

We can take the derivative of a function that indirectly depends on $x$, by taking the derivative of the outer function and multiplying it by the derivative of the inner function, or

$f'(x) = f'(g(x))*g'(x)$

### Try it again

Let's go through some more examples.

$$ f(x) = (3x^2 + 10x)^3$$

> Stop here, and give this a shot on your own.  The answer will always be waiting for you right below, so you really have nothing to lose.  No one will know if you struggle - and it's kinda the point.

**1. Divide the function into two components **

$$g(x) = 3x^2 + 10x $$
$$f(x) = (g(x))^3$$


** 2. Take the derivative of each of the component functions **

$$g'(x) = 6x + 10 $$
$$f'(x) = 3g(x))^2$$

** 3. Substitution **

$$f'(x) = f'(g(x))*g'(x) = 3(g(x))^2*(6x+10)$$

Then substituting in $g(x) = 3x^2 + 10x $ we have:

$$f'(x) = 3*(3x^2 + 10x)^2*(6x+10) $$

And we can leave it there for now.

### Summary

In this lesson, we learned about the chain rule.  The chain rule allows us to take the derivative of a function that that comprises of another function that depends on $x$.  We apply the chain by taking the derivative of the outer function and multiplying that by the derivative of the inner function.  We'll see the chain rule in the future when in our work with gradient descent.
