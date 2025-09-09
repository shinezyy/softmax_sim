# RISC-V Vector Processor Softmax Simulator

A detailed microarchitectural simulator for modeling RISC-V vector processor execution of softmax operations with configurable architecture parameters and instruction scheduling policies.

## Overview

This simulator models a RISC-V vector processor executing softmax computations with support for:
- **Execution Modes**: In-order and out-of-order instruction scheduling
- **Configurable Architecture**: Variable register widths, compute unit widths, and cache bandwidth
- **Instruction Chaining**: Pipeline optimization through element-wise data forwarding
- **Detailed Modeling**: Instruction decomposition into micro-operations (μops) with cycle-accurate simulation
- **Visualization**: ASCII-based timeline visualization of instruction and μop execution

## Architecture Features

### Processor Configuration
- **Register Width**: 512, 1024, or 2048 bits
- **Compute Unit Widths**: Separate widths for different operation types
  - Reduce operations: 128, 256, 512, or 1024 bits
  - Simple elementwise operations (FMA): 128, 256, 512, or 1024 bits  
  - Complex elementwise operations (EXP2): 128, 256, 512, or 1024 bits
- **Cache Bandwidth**: 32, 64, or 128 bytes per cycle
- **Execution Modes**: In-order vs out-of-order with configurable window size

### Supported Instructions
1. **REDUCE**: Computes maximum of M bf16 elements (M = register_width/16)
2. **FMA**: Element-wise fused multiply-add on bf16 elements
3. **LOAD**: Memory load operations with cache bandwidth constraints
4. **STORE**: Memory store operations sharing cache bandwidth with loads
5. **EXP2**: Element-wise exponential (base 2) operations on bf16 elements

### Instruction Chaining
When enabled, chaining allows dependent instructions to begin execution as soon as partial results from producer instructions become available, improving pipeline utilization. This requires both producer and consumer instructions to have element-wise data patterns.

## Installation

No special installation required. The simulator is a single Python file with standard library dependencies only.

```bash
git clone <repository-url>
cd softmax_sim
```

## Usage

### Command Line Interface

```bash
python softmax_simulator.py [options]
```

### Available Options

```
usage: softmax_simulator.py [-h] [--execution-mode {in-order,out-of-order}]
                            [--register-width {512,1024,2048}]
                            [--reduce-compute-width {128,256,512,1024}]
                            [--simple-elementwise-width {128,256,512,1024}]
                            [--complex-elementwise-width {128,256,512,1024}]
                            [--cache-bandwidth {32,64,128}] [--chaining]
                            [--ooo-window-size OOO_WINDOW_SIZE]

RISC-V Vector Processor Softmax Simulator

options:
  -h, --help            show this help message and exit
  --execution-mode {in-order,out-of-order}
                        Execution mode for the processor (default: in-order)
  --register-width {512,1024,2048}
                        Register width in bits (default: 2048)
  --reduce-compute-width {128,256,512,1024}
                        Reduce compute unit width in bits (default: 512)
  --simple-elementwise-width {128,256,512,1024}
                        Simple elementwise compute unit width in bits (default: 512)
  --complex-elementwise-width {128,256,512,1024}
                        Complex elementwise compute unit width in bits (default: 512)
  --cache-bandwidth {32,64,128}
                        Cache bandwidth in bytes per cycle (default: 64)
  --chaining            Enable chaining (default: True)
  --ooo-window-size OOO_WINDOW_SIZE
                        Out-of-order execution window size (default: 128)
```

### Example Usage

```bash
# Run with default in-order configuration
python softmax_simulator.py

# Run out-of-order execution with larger window
python softmax_simulator.py --execution-mode out-of-order --ooo-window-size 256

```

## Output Format

The simulator provides detailed execution analysis including:

### Execution Summary
- Total execution cycles
- Instruction timeline with issue, start, and completion cycles
- Micro-operation breakdown and timing
- Performance metrics (instructions per cycle)

### ASCII Visualization
The simulator generates timeline visualizations using ASCII characters:
- `@`: Instruction/μop issue cycle
- `-`: Execution in progress  
- `!`: Completion cycle

Example visualization:
```
Instruction Timeline Visualization:
Instruction 0 (load):     @---!
Instruction 1 (fma):       @--!
Instruction 2 (exp2):        @----!
Instruction 3 (reduce):        @------!

Micro-operation Timeline Visualization:
uop 0.0 (load):      @---!
uop 1.0 (fma):        @--!
uop 2.0 (exp2):         @----!
uop 3.0 (reduce):         @--!
uop 3.1 (reduce):           @--!
```

## Architecture Details

### Instruction Decomposition
Instructions are decomposed into micro-operations based on data size and compute unit width:
- **Element-wise operations**: Decomposed by compute unit width (e.g., 2048-bit register with 512-bit compute → 4 μops)
- **Reduce operations**: Multi-stage reduction tree with dependencies between stages
- **Memory operations**: Limited by cache bandwidth constraints

### Dependency Modeling  
The simulator tracks both:
- **Instruction-level dependencies**: Specified in the instruction stream
- **μop-level dependencies**: Generated during instruction decomposition
- **Resource constraints**: Compute unit availability and cache bandwidth

### Chaining Implementation
When chaining is enabled:
- Producer instructions mark partial results as ready
- Consumer instructions can begin processing available elements
- Requires matching μop counts between chained instructions
- Only applies to element-wise instruction patterns

## Development History

The simulator was developed iteratively with the following key features added:
1. **Initial Architecture**: Basic instruction modeling and execution simulation
2. **Visualization**: ASCII timeline visualization for instructions and micro-operations  
3. **Chaining Support**: Implementation of instruction chaining for pipeline optimization
4. **Arithmetic Limits**: Proper resource constraint modeling for different instruction types
5. **Instruction Types**: Separation of compute resources by operation complexity
6. **Command Line Interface**: Argparse integration for configuration flexibility

## Performance Characteristics

The simulator models realistic processor behavior including:
- **Pipeline effects**: Multi-cycle instruction latencies with pipelining
- **Resource contention**: Shared cache bandwidth between loads and stores
- **Execution policies**: Different scheduling behaviors for in-order vs out-of-order
- **Data forwarding**: Chaining optimization for compatible instruction sequences

## Files

- `softmax_simulator.py`: Main simulator implementation
- `prompts_used_in_cursor_for_softmax_sim.txt`: Development history and design requirements
- `README.md`: This documentation file

## Contributing

The simulator architecture supports extension for:
- Additional instruction types
- Different processor configurations  

---

*This simulator was designed to evaluate microarchitectural trade-offs in RISC-V vector processors for machine learning workloads, with particular focus on softmax computation patterns common in transformer architectures.*
