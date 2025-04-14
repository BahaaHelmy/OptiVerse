
# 🌌 OptiVerse

**OptiVerse** is a Python library that integrates a universe of metaheuristic optimization algorithms. Whether you're solving mathematical, engineering, or machine learning optimization problems, OptiVerse provides powerful, customizable, and easy-to-use implementations of nature-inspired algorithms.

## 🚀 Features

- ✅ A collection of population-based and swarm intelligence algorithms
- 🧩 Clean and modular codebase for easy customization
- 📈 Built-in convergence tracking
- 🧪 Simple integration with custom objective functions
- 🐍 Lightweight with minimal dependencies

## 📦 Installation

Install the latest version of OptiVerse from PyPI:

```bash
pip install OptiVerse
```

Or install it directly from GitHub:

```bash
pip install git+https://github.com/BahaaHelmy/OptiVerse.git
```

## 🧠 Algorithms Included

- BAT Algorithm (BAT)
- (Add more here as you expand: PSO, GWO, DE, etc.)

## 🧰 Usage

Here's how to use the BAT algorithm from OptiVerse:

```python
from OptiVerse.BAT import BAT

# Define your objective function
def sphere(x):
    return sum(i**2 for i in x)

# Run the BAT algorithm
result = BAT(sphere, -10, 10, 30, 50, 100)

# Print the best solution found
print("Best solution:", result.best)
```

### Parameters:
- `sphere`: Objective function to minimize
- `-10`, `10`: Lower and upper bounds
- `30`: Number of dimensions
- `50`: Population size
- `100`: Number of iterations

The `result` object also contains:
- `result.convergence`: A list tracking the best fitness value per iteration
- `result.executionTime`: Total time taken for optimization
- `result.objfname`: Name of the objective function

## 📊 Convergence Curve (Optional)

You can visualize the convergence of the algorithm like this:

```python
import matplotlib.pyplot as plt

plt.plot(result.convergence)
plt.title("Convergence Curve")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.grid()
plt.show()
```

## 🤝 Contributing

We welcome contributions from the community! If you’d like to add new algorithms, improve documentation, or fix bugs, feel free to:

1. Fork this repo
2. Create a new branch (`git checkout -b feature-xyz`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push and create a pull request

---

## 🌠 Author

Created with ❤️ by Bahaa Helmy  
Feel free to reach out on [LinkedIn](https://www.linkedin.com/in/bahaahelmy/](https://www.linkedin.com/in/bahaa-el-din-helmy-ph-d-867108116/)) or contribute on [GitHub](https://github.com/BahaaHelmy).
