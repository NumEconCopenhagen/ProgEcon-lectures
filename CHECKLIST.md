# Learning checklist

The three lists below are the official learning outcomes from the
[course description](DESCRIPTION.md). Under each one are the concrete things you should be able to
do.

---

## Knowledge

### Describe the differences between fundamental data types

- [ ] I can describe what `int`, `float`, `str` and `bool` are for, convert between them, and check
      with `type()`.
- [ ] I can use the arithmetic, comparison and logical operators, and augmented assignment.
- [ ] I can explain why `0.1 + 0.2 == 0.3` is `False`, and use `np.isclose` instead.

### Describe the differences between data containers

- [ ] I can say what **lists**, **tuples**, **dictionaries** and **arrays** are each good at, and
      which are mutable.
- [ ] I can index and slice with 0-based indexing, negative indices and `[a:b:step]`.
- [ ] I can index arrays with boolean masks and index lists, and reduce along an axis
      (`.sum(axis=0)`, `argmin`, ...).
- [ ] I can explain **broadcasting** and use it to evaluate a function on a grid without loops.
- [ ] I know the difference between `A*B` and `A@B`.

### Explain the use of conditionals

- [ ] I can write `if` / `elif` / `else`.
- [ ] I know what counts as `True` for non-boolean types.

### Explain the use of loops

- [ ] I can write `for` and `while` loops and nest them.
- [ ] I can use `enumerate` and `zip`, and I know when `continue` and `break` are the right tool.
- [ ] I know which parts of a problem *must* be a loop (a recursion over time) and which should be
      vectorized instead.

### Explain the use of functions, methods and classes

- [ ] I can write a function with several inputs and outputs, keyword arguments and default values.
- [ ] **I can explain scope**, and why relying on a global variable inside a function is dangerous.
- [ ] I know a function is an object that can be passed to another function
- [ ] I can write a class with `__init__`, attributes and methods, explain `self`, and create
      several instances.
- [ ] I can add operator methods (`__str__`).

### Describe the difference between views and copies of objects

- [ ] **I can predict when changing one variable changes another** — reference vs. copy vs. deep copy.
- [ ] **I know which numpy operations give a view and which give a copy**, and what goes wrong if I
      confuse them.

### Explain how to use numerical optimizers and root-finders

- [ ] I can solve a problem by **brute-force grid search**, and say what limits its accuracy.
- [ ] I understand an optimizer as **a loop with two stopping rules**: a tolerance and a maximum
      number of iterations.
- [ ] I can explain and implement **Newton's method** and **bisection**, and say what each requires.
- [ ] I can compute a numerical derivative and choose a sensible step size.
- [ ] I know a function can have several roots or optima, and that the starting point or bracket
      decides which one I get.

### Explain how to use (pseudo) random numbers

- [ ] I can explain why a **seed** makes results reproducible, and what a generator's state is.
- [ ] I can explain a **Monte Carlo** calculation, and say how the error shrinks with the number of
      draws.

---

## Skills

### Setup a Python environment

- [ ] I have a working installation and can open a workspace folder in VS Code.
- [ ] I can explain what the interpreter, the kernel, a package and an environment are, and restart
      the kernel and run all cells.
- [ ] I can install a package that is missing.

### Write Python scripts, functions and notebooks

- [ ] I can explain the difference between a script, a notebook and a module.
- [ ] I can write my own module and `import` it (and use `%autoreload 2`).

### Structure and document code

- [ ] My project lives in **one folder** with **one file that runs it all**; long functions and classes go
      in `.py` modules, the notebook calls them.
- [ ] I can write **line comments** and **docstrings**
- [ ] I follow a consistent style: e.g. 4-space indentation, `CamelCase` classes, `lower_case` functions
      and variables, ordered section comments.
- [ ] My code **limits repeated code-lines**, is split into small functions, and has no unexpected
      side effects.

### Test and debug code

- [ ] I read a traceback from the bottom and use printing to investigate an error
- [ ] I write `assert` statements on cases I already know the answer to.
- [ ] **I can test with a property rather than a known answer** — something that must hold whatever
      the answer turns out to be.

### Use a version control system (Git)

*This is not required material*

- [ ] I can explain local vs. remote repository, branch and `.gitignore`.
- [ ] I know what fetch, merge, pull, stage, commit and push mean.
- [ ] I can *commit all* and *sync* from VS Code, write a useful commit message, and inspect what
      changed before committing.
- [ ] I can resolve a merge conflict — and I know syncing before I start work prevents most of them.

### Import and export data and use online databases (APIs)

- [ ] I can read messy Excel and CSV files, and write a cleaned data set back to file.
- [ ] I can download from an **API** (`dstapi`, FRED) and **verify** the result against the source.
- [ ] I can write output to a text file and read it back in.
- [ ] I can save and load results with `pickle` and `np.savez` / `np.load`.

### Summarize and visualize data

- [ ] I can build and inspect a `DataFrame` (`.head`, `.describe`, `.info`, `.dtypes`).
- [ ] I can select by name (`.loc`), by position (`.iloc`) and by condition, and add, change, drop
      and rename variables.
- [ ] I can tell **long** and **wide** data apart and convert between them (`pivot_table`, `melt`).
- [ ] I can `concat` and `merge`, say what an **inner** and a **left join** do to the number of rows,
      and check for unmatched rows afterwards.
- [ ] I can use `.groupby` with `.agg`, and `.transform` when I need to keep the shape of the data.
- [ ] I can format numbers with f-strings (`f'{x:8.3f}'`) and print a readable table.
- [ ] I can make a 2D plot with the `fig, ax` interface — labels, title, legend, grid — plus several
      lines, several panels, and a 3D surface, and save it with `fig.savefig`.

### Use numerical optimizers and equation solvers

- [ ] I can call `optimize.minimize_scalar` and `optimize.minimize`, and I know the first hands my
      function a *number* and the second an *array*.
- [ ] I know of `Nelder-Mead`, `BFGS`, `L-BFGS-B` and `SLSQP`, and their basic differences.
- [ ] I can pass parameters to an objective with `args=(...)` and a `lambda` function.
- [ ] I can handle a constraint by substituting it away, re-parameterizing, or giving it to a solver
      that accepts constraints.
- [ ] I can use `optimize.root_scalar` and `optimize.root`, including on a system of equations.
- [ ] **I know `success = True` is not a proof** — I check the constraints, compare methods, and use
      several starting points.

### Solve models numerically

- [ ] I can put a model in a class: parameters as attributes, equations and solvers as methods,
      results collected in e.g. `self.sol` or `self.sim`.
- [ ] I can solve the same problem **by hand and numerically**, and use the first to test the second.
- [ ] I can write methods with plain arithmetic so the same code runs on one input or a whole grid.
- [ ] When I write my own algorithm, **I return whether it converged** — and I check it.

### Simulate models

- [ ] I can create a generator with `np.random.default_rng(seed)`, draw from the standard
      distributions in vectorized form, and use several generators side by side.
- [ ] I can simulate many units at once — **vectorizing over the cross-section and looping only over
      time**.
- [ ] I can let an argument be **either a number or an array**, so one method covers the baseline, a
      permanent change and random shocks.
- [ ] I can solve many random versions of a model and summarize them with percentiles rather than a
      single number.

### Calibrate economic models to data
 
*This is not required material*

### Calibrate economic models to data
 
*This is not required material*

- [ ] I can explain what data moments or targets are being matched in a calibration.
- [ ] I can write an objective function measuring the distance between simulated and empirical moments.
- [ ] I can calibrate model parameters with a numerical optimizer and impose economically meaningful bounds.
- [ ] I can check that the calibrated parameters reproduce the target moments.
- [ ] I can interpret the calibrated parameters and explain which targets identify them.
- [ ] I can assess sensitivity to starting values, parameter bounds and alternative targets.

---

## Competencies

### Write well-structured and well-documented code

- [ ] My project runs top to bottom after *Restart kernel and run all cells* — on another person's
      computer.
- [ ] Someone who did not write my code can read it and understand what it does and why.
- [ ] My figures and tables can be understood on their own.

### Work collaboratively on code projects

- [ ] We split the work so that two people can work at the same time without constant conflicts.

### Present and discuss results of a numerical analysis

- [ ] The notebook reads as a report: every result is introduced, explained and interpreted in words.
- [ ] I can present and defend the results orally — including what I chose *not* to do, and why.
- [ ] I can take provided class code, understand its structure, and extend it.
- [ ] I can decide whether a question calls for an optimizer, a root-finder, a simulation or
      pandas — and I sanity-check numerical answers.
