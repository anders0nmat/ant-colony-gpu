# Ant-Colony Optimization on dedicated GPUs

This is a research project to build an Ant-Colony Optimization (ACO),
first as a sequential algorithm on the CPU,
second as a parallelized algorithm on the GPU with OpenCL.
The algoroithm attempts to solve a Sequential Ordering Problem.

## Order of operation

- [x] Build a sequential ACO algorithm
   - [x] Build a Graph representation
   - [x] Build a Problem Reader (for TSPLIB-Problems)
   - [x] Build a framework for easily switching variants of the optimization
   - [ ] Build a framework for profiling and generating performance data
- [ ] Parallelize the algorithm with OpenCL
   - [ ] Find a memory representation of the problem, that fits the needs of GPUs
   - [ ] Build the easiest possible parallelization
- [ ] Optimize the algorithm
   - [ ] Build new variants with different strategies
   - [ ] Measure their performance
- [ ] Evaluate the results
   - [ ] What worked well, what not so much? Why?

## Scientific sources

[OpenCL API Specification](https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html)
