#!/usr/bin/env python3
"""
RISC-V Vector Processor Softmax Simulator

This simulator models a RISC-V vector processor executing softmax operations
with configurable architecture parameters and instruction scheduling.
"""

from enum import Enum
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
import math
import copy


class InstructionType(Enum):
    REDUCE = "reduce"
    FMA = "fma" 
    LOAD = "load"
    STORE = "store"
    EXP2 = "exp2"


class ExecutionMode(Enum):
    IN_ORDER = "in_order"
    OUT_OF_ORDER = "out_of_order"


@dataclass
class ProcessorConfig:
    """Configuration parameters for the RISC-V vector processor"""
    register_width: int  # rl: 512, 1024, 2048 bits
    compute_unit_width: int  # vl: 128, 256, 512, 1024 bits (must be <= register_width)
    cache_bandwidth: int  # 32, 64, 128 bytes per cycle
    execution_mode: ExecutionMode
    chaining_enabled: bool = False
    chaining_granularity: int = 32  # 32, 64, 128, 256 bytes
    
    # Instruction latencies
    reduce_latency: int = 7
    fma_latency: int = 4
    load_latency: int = 10
    store_latency: int = 10
    exp2_latency: int = 15
    
    # Out-of-order execution window size
    ooo_window_size: int = 16
    
    def __post_init__(self):
        if self.compute_unit_width > self.register_width:
            raise ValueError("Compute unit width cannot exceed register width")
        
        valid_reg_widths = [512, 1024, 2048]
        valid_compute_widths = [128, 256, 512, 1024]
        valid_cache_bw = [32, 64, 128]
        valid_chain_gran = [32, 64, 128, 256]
        
        if self.register_width not in valid_reg_widths:
            raise ValueError(f"Invalid register width: {self.register_width}")
        if self.compute_unit_width not in valid_compute_widths:
            raise ValueError(f"Invalid compute unit width: {self.compute_unit_width}")
        if self.cache_bandwidth not in valid_cache_bw:
            raise ValueError(f"Invalid cache bandwidth: {self.cache_bandwidth}")
        if self.chaining_granularity not in valid_chain_gran:
            raise ValueError(f"Invalid chaining granularity: {self.chaining_granularity}")


@dataclass
class Instruction:
    """Represents a single instruction in the instruction stream"""
    id: int
    type: InstructionType
    dependencies: Set[int]  # IDs of instructions this depends on
    data_size: int  # Size of data to process (in bytes)
    target_register: Optional[int] = None
    
    # Chaining support
    element_wise_src: bool = False
    element_wise_dest: bool = False
    
    # Execution state
    issued: bool = False
    started: bool = False
    completed: bool = False
    issue_cycle: int = -1  # When instruction was issued
    start_cycle: int = -1  # When execution started
    complete_cycle: int = -1


# Instruction wrapper classes for simplified instruction creation
class LoadInstruction:
    """Simplified wrapper for creating LOAD instructions"""
    
    def __init__(self, id: int, target_register: int, dependencies: Set[int] = None, data_size: int = 256):
        self.instruction = Instruction(
            id=id,
            type=InstructionType.LOAD,
            dependencies=dependencies or set(),
            data_size=data_size,  # Default 2048 bits = 256 bytes
            target_register=target_register,
            element_wise_dest=True,
        )
    
    def __getattr__(self, name):
        return getattr(self.instruction, name)


class ReduceInstruction:
    """Simplified wrapper for creating REDUCE instructions"""
    
    def __init__(self, id: int, target_register: int, source_registers: List[int], 
                 dependencies: Set[int] = None, data_size: int = 256):
        # If dependencies not explicitly provided, derive from source_registers
        if dependencies is None:
            dependencies = set(source_registers) if source_registers else set()
        
        self.instruction = Instruction(
            id=id,
            type=InstructionType.REDUCE,
            dependencies=dependencies,
            data_size=data_size,  # Default 2048 bits = 256 bytes
            target_register=target_register,
            element_wise_src=True,
        )
        # Reduce instruction specific
        self.first_level_uop_count: int = 0  # Number of first-level uops for reduce instructions
        
    
    def __getattr__(self, name):
        return getattr(self.instruction, name)


class FMAInstruction:
    """Simplified wrapper for creating FMA instructions"""
    
    def __init__(self, id: int, target_register: int, source_registers: List[int],
                 dependencies: Set[int] = None, data_size: int = 256):
        # If dependencies not explicitly provided, derive from source_registers
        if dependencies is None:
            dependencies = set(source_registers) if source_registers else set()
        
        self.instruction = Instruction(
            id=id,
            type=InstructionType.FMA,
            dependencies=dependencies,
            data_size=data_size,  # Default 2048 bits = 256 bytes
            target_register=target_register,
            element_wise_src=True,
            element_wise_dest=True,
        )
    
    def __getattr__(self, name):
        return getattr(self.instruction, name)


class EXP2Instruction:
    """Simplified wrapper for creating EXP2 instructions"""
    
    def __init__(self, id: int, target_register: int, source_registers: List[int],
                 dependencies: Set[int] = None, data_size: int = 256):
        # If dependencies not explicitly provided, derive from source_registers
        if dependencies is None:
            dependencies = set(source_registers) if source_registers else set()
        
        self.instruction = Instruction(
            id=id,
            type=InstructionType.EXP2,
            dependencies=dependencies,
            data_size=data_size,  # Default 2048 bits = 256 bytes
            target_register=target_register,
            element_wise_src=True,
            element_wise_dest=True,
        )
    
    def __getattr__(self, name):
        return getattr(self.instruction, name)


class StoreInstruction:
    """Simplified wrapper for creating STORE instructions"""
    
    def __init__(self, id: int, source_registers: List[int], 
                 dependencies: Set[int] = None, data_size: int = 256):
        # If dependencies not explicitly provided, derive from source_registers
        if dependencies is None:
            dependencies = set(source_registers) if source_registers else set()
        
        self.instruction = Instruction(
            id=id,
            type=InstructionType.STORE,
            dependencies=dependencies,
            data_size=data_size,  # Default 2048 bits = 256 bytes
            element_wise_src=True,
        )
    
    def __getattr__(self, name):
        return getattr(self.instruction, name)


@dataclass 
class MicroOp:
    """Represents a micro-operation (uop) - a unit of work that can be executed"""
    instruction_id: int
    uop_id: int
    type: InstructionType
    data_size: int  # Size of data this uop processes
    dependencies: Set[int]  # Other uops this depends on
    latency: int
    
    # Execution state
    issued: bool = False
    started: bool = False
    completed: bool = False
    start_cycle: int = -1
    complete_cycle: int = -1
    
    # Chaining support - tracks how many elements are ready
    ready_elements: int = 0


class InstructionExecutor:
    """Handles the execution logic for different instruction types"""
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
    
    def split_instruction_to_uops(self, instruction: Instruction) -> List[MicroOp]:
        """Split an instruction into micro-operations based on processor configuration"""
        uops = []
        
        if instruction.type == InstructionType.REDUCE:
            uops = self._split_reduce_instruction(instruction)
        elif instruction.type in [InstructionType.FMA, InstructionType.EXP2]:
            uops = self._split_arithmetic_instruction(instruction)
        elif instruction.type in [InstructionType.LOAD, InstructionType.STORE]:
            uops = self._split_memory_instruction(instruction)
        
        return uops
    
    def _split_reduce_instruction(self, instruction: Instruction) -> List[MicroOp]:
        """Split reduce instruction into uops with tree reduction logic"""
        # Max elements per instruction: M = rl/16 (bf16 = 2 bytes, so 16 bits)
        max_elements = self.config.register_width // 16
        # Elements per cycle: N = vl/16  
        elements_per_cycle = self.config.compute_unit_width // 16
        
        # Actual elements to process
        actual_elements = min(max_elements, instruction.data_size // 2)  # bf16 = 2 bytes
        
        uops = []
        uop_id = 0
        
        # Phase 1: Parallel reduction of groups
        if actual_elements <= elements_per_cycle:
            # Single uop can handle all elements
            uop = MicroOp(
                instruction_id=instruction.id,
                uop_id=uop_id,
                type=InstructionType.REDUCE,
                data_size=actual_elements * 2,  # bf16 elements
                dependencies=set(),
                latency=self.config.reduce_latency
            )
            uops.append(uop)
            instruction.first_level_uop_count = 1
        else:
            # Multiple uops needed for first phase
            first_phase_uops = math.ceil(actual_elements / elements_per_cycle)
            
            for i in range(first_phase_uops):
                elements_in_uop = min(elements_per_cycle, 
                                    actual_elements - i * elements_per_cycle)
                uop = MicroOp(
                    instruction_id=instruction.id,
                    uop_id=uop_id,
                    type=InstructionType.REDUCE,
                    data_size=elements_in_uop * 2,
                    dependencies=set(),
                    latency=self.config.reduce_latency
                )
                uops.append(uop)
                uop_id += 1
            instruction.first_level_uop_count = first_phase_uops
            
            # Phase 2: Reduce the results from phase 1
            if first_phase_uops > 1:
                # Create dependency on all first phase uops
                first_phase_deps = set(range(len(uops)))
                
                # Calculate how many reduction levels are needed
                remaining_elements = first_phase_uops
                while remaining_elements > 1:
                    next_level_uops = math.ceil(remaining_elements / elements_per_cycle)
                    level_deps = first_phase_deps.copy()
                    
                    for i in range(next_level_uops):
                        uop = MicroOp(
                            instruction_id=instruction.id,
                            uop_id=uop_id,
                            type=InstructionType.REDUCE,
                            data_size=min(elements_per_cycle, remaining_elements) * 2,
                            dependencies=level_deps,
                            latency=self.config.reduce_latency
                        )
                        uops.append(uop)
                        uop_id += 1
                    
                    remaining_elements = next_level_uops
                    first_phase_deps = set(range(len(uops) - next_level_uops, len(uops)))
        
        return uops
    
    def _split_arithmetic_instruction(self, instruction: Instruction) -> List[MicroOp]:
        """Split FMA or EXP2 instruction into uops"""
        # Max elements per instruction: rl/16 (bf16)
        max_elements = self.config.register_width // 16
        # Elements per cycle: vl/16
        elements_per_cycle = self.config.compute_unit_width // 16
        
        actual_elements = min(max_elements, instruction.data_size // 2)  # bf16 = 2 bytes
        
        uops = []
        remaining_elements = actual_elements
        uop_id = 0
        
        while remaining_elements > 0:
            elements_in_uop = min(elements_per_cycle, remaining_elements)
            
            latency = (self.config.fma_latency if instruction.type == InstructionType.FMA 
                      else self.config.exp2_latency)
            
            uop = MicroOp(
                instruction_id=instruction.id,
                uop_id=uop_id,
                type=instruction.type,
                data_size=elements_in_uop * 2,
                dependencies=set(),
                latency=latency
            )
            uops.append(uop)
            
            remaining_elements -= elements_in_uop
            uop_id += 1
        
        return uops
    
    def _split_memory_instruction(self, instruction: Instruction) -> List[MicroOp]:
        """Split load or store instruction into uops"""
        # Max bytes per instruction: vl/8
        max_bytes_per_instruction = self.config.register_width // 8
        # Bytes per cycle limited by cache bandwidth
        bytes_per_cycle = self.config.cache_bandwidth
        actual_bytes = instruction.data_size // 8

        print(f"Max bytes per instruction: {max_bytes_per_instruction}")
        print(f"Bytes per cycle: {bytes_per_cycle}")
        print(f"Instruction data bytes: {actual_bytes}")

        assert actual_bytes <= max_bytes_per_instruction
        
        uops = []
        remaining_bytes = actual_bytes
        uop_id = 0
        
        while remaining_bytes > 0:
            bytes_in_uop = min(bytes_per_cycle, remaining_bytes)
            
            latency = (self.config.load_latency if instruction.type == InstructionType.LOAD
                      else self.config.store_latency)
            
            # Create dependencies: each uop depends on the previous one (except the first)
            dependencies = set()
            
            uop = MicroOp(
                instruction_id=instruction.id,
                uop_id=uop_id,
                type=instruction.type,
                data_size=bytes_in_uop,
                dependencies=dependencies,
                latency=latency
            )
            # print(f"uop {uop_id} dependencies: {dependencies}")
            uops.append(uop)
            
            remaining_bytes -= bytes_in_uop  
            uop_id += 1
        
        return uops


class VectorProcessor:
    """Main vector processor simulator"""
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.executor = InstructionExecutor(config)
        
        # Simulation state
        self.current_cycle = 0
        self.instructions: List[Instruction] = []
        self.uops: List[MicroOp] = []
        self.instruction_uop_map: Dict[int, List[int]] = {}  # instruction_id -> uop_ids
        
        # Execution units (simplified model)
        self.execution_units = {
            InstructionType.REDUCE: [],
            InstructionType.FMA: [],
            InstructionType.EXP2: [],
            InstructionType.LOAD: [],
            InstructionType.STORE: []
        }
        
        # Cache bandwidth tracking
        self.cache_bandwidth_used = 0
    
    def load_instructions(self, instructions: List[Instruction]):
        """Load instruction stream into the processor"""
        self.instructions = instructions
        self.uops = []
        self.instruction_uop_map = {}
        
        # Split all instructions into uops
        uop_offset = 0
        for instruction in self.instructions:
            instruction_uops = self.executor.split_instruction_to_uops(instruction)
            self.uops.extend(instruction_uops)
            
            uop_ids = list(range(uop_offset, uop_offset + len(instruction_uops)))
            self.instruction_uop_map[instruction.id] = uop_ids
            uop_offset += len(instruction_uops)
        
        # Fix reduce instruction dependencies
        self._fix_reduce_dependencies()
        
        # Fix memory instruction dependencies
        self._fix_memory_dependencies()
        
        # Establish chaining dependencies if enabled
        if self.config.chaining_enabled:
            self._establish_chaining_dependencies()
    
    def _fix_reduce_dependencies(self):
        """Fix dependencies for reduce instruction uops after all uops are loaded"""
        for instruction in self.instructions:
            if instruction.type == InstructionType.REDUCE:
                uop_ids = self.instruction_uop_map[instruction.id]
                if len(uop_ids) > 1:  # Multiple uops for this reduce instruction
                    # Find the boundary between first phase and subsequent phases
                    # First, calculate expected first phase uops
                    max_elements = self.config.register_width // 16
                    elements_per_cycle = self.config.compute_unit_width // 16
                    actual_elements = min(max_elements, instruction.data_size // 2)
                    
                    if actual_elements > elements_per_cycle:
                        first_phase_uops = math.ceil(actual_elements / elements_per_cycle)
                        
                        # Clear existing dependencies for all uops of this instruction
                        for uop_id in uop_ids:
                            self.uops[uop_id].dependencies.clear()
                        
                        # Set correct dependencies: second phase uops depend on all first phase uops
                        if len(uop_ids) > first_phase_uops:
                            first_phase_global_ids = set(uop_ids[:first_phase_uops])
                            for i in range(first_phase_uops, len(uop_ids)):
                                uop_global_id = uop_ids[i]
                                self.uops[uop_global_id].dependencies = first_phase_global_ids.copy()
    
    def _fix_memory_dependencies(self):
        """Fix dependencies for memory instruction uops after all uops are loaded"""
        for instruction in self.instructions:
            if instruction.type in [InstructionType.LOAD, InstructionType.STORE]:
                uop_ids = self.instruction_uop_map[instruction.id]
                
                # Fix dependencies: convert local IDs to global IDs
                for i, global_uop_id in enumerate(uop_ids):
                    uop = self.uops[global_uop_id]
                    new_dependencies = set()
                    
                    for local_dep_id in uop.dependencies:
                        # Convert local dependency ID to global ID
                        global_dep_id = uop_ids[local_dep_id]
                        new_dependencies.add(global_dep_id)
                    
                    uop.dependencies = new_dependencies
    
    def _establish_chaining_dependencies(self):
        """Establish chaining dependencies between producer and consumer instructions"""
        for producer_inst in self.instructions:
            # Skip if producer doesn't have element-wise destination
            if not producer_inst.element_wise_dest:
                continue
            
            # Find consumer instructions that depend on this producer
            for consumer_inst in self.instructions:
                # Skip if consumer doesn't have element-wise source
                if not consumer_inst.element_wise_src:
                    continue
                
                # Skip if consumer doesn't depend on producer
                if producer_inst.id not in consumer_inst.dependencies:
                    continue
                
                # Get uops for both instructions
                producer_uop_ids = self.instruction_uop_map[producer_inst.id]
                consumer_uop_ids = self.instruction_uop_map[consumer_inst.id]
                
                # Assert that data sizes match
                assert producer_inst.data_size == consumer_inst.data_size, \
                    f"Chaining requires matching data sizes: producer {producer_inst.id} " \
                    f"({producer_inst.data_size}) vs consumer {consumer_inst.id} " \
                    f"({consumer_inst.data_size})"
                
                # Assert that number of uops match
                if consumer_inst.type == InstructionType.REDUCE:
                    assert len(producer_uop_ids) == consumer_inst.first_level_uop_count, \
                        f"Chaining requires matching uop counts: producer {producer_inst.id} " \
                        f"({len(producer_uop_ids)} uops) vs consumer {consumer_inst.id} " \
                        f"({consumer_inst.first_level_uop_count} uops)"
                else:
                    assert len(producer_uop_ids) == len(consumer_uop_ids), \
                        f"Chaining requires matching uop counts: producer {producer_inst.id} " \
                        f"({len(producer_uop_ids)} uops) vs consumer {consumer_inst.id} " \
                        f"({len(consumer_uop_ids)} uops)"
                
                # Assert for load instructions - they should match compute instruction uop count
                if producer_inst.type == InstructionType.LOAD:
                    # Find the next compute instruction that depends on this load
                    for next_inst in self.instructions:
                        if (producer_inst.id in next_inst.dependencies and 
                            next_inst.type in [InstructionType.FMA, InstructionType.EXP2]):
                            next_uop_count = len(self.instruction_uop_map[next_inst.id])
                            assert len(producer_uop_ids) == next_uop_count, \
                                f"Load instruction {producer_inst.id} uop count ({len(producer_uop_ids)}) " \
                                f"must match compute instruction {next_inst.id} uop count ({next_uop_count})"
                            break
                
                # Establish one-to-one chaining dependencies
                print(f"Establishing chaining between instruction {producer_inst.id} -> {consumer_inst.id}")
                for i, (prod_uop_id, cons_uop_id) in enumerate(zip(producer_uop_ids, consumer_uop_ids)):
                    # Consumer uop depends on corresponding producer uop completion
                    self.uops[cons_uop_id].dependencies.add(prod_uop_id)
                    print(f"  uop {consumer_inst.id}.{i} now depends on uop {producer_inst.id}.{i}")
                
                # Remove instruction-level dependency since we now have uop-level dependencies
                consumer_inst.dependencies.discard(producer_inst.id)
                print(f"  Removed instruction-level dependency {producer_inst.id} -> {consumer_inst.id}")
    
    def simulate(self, max_cycles: int = 10000) -> Dict:
        """Run the simulation and return results"""
        self.current_cycle = 0
        
        # Reset all state
        for instruction in self.instructions:
            instruction.issued = instruction.started = instruction.completed = False
            instruction.issue_cycle = instruction.start_cycle = instruction.complete_cycle = -1
        
        for uop in self.uops:
            uop.issued = uop.started = uop.completed = False
            uop.start_cycle = uop.complete_cycle = -1
            uop.ready_elements = 0
        
        while not self._all_instructions_completed() and self.current_cycle < max_cycles:
            self._simulate_cycle()
            self.current_cycle += 1
        
        if self.current_cycle >= max_cycles:
            print(f"Warning: Simulation reached maximum cycles ({max_cycles})")
            print(f"Completed instructions: {sum(1 for inst in self.instructions if inst.completed)}/{len(self.instructions)}")
            print(f"Completed uops: {sum(1 for uop in self.uops if uop.completed)}/{len(self.uops)}")
        
        return self._generate_results()
    
    def _simulate_cycle(self):
        """Simulate a single cycle"""
        # Reset per-cycle state
        self.cache_bandwidth_used = 0
        
        # Update execution units and complete uops
        self._update_execution_units()
        
        # Issue new uops based on execution mode
        if self.config.execution_mode == ExecutionMode.IN_ORDER:
            self._issue_in_order()
        else:
            self._issue_out_of_order()
        
        # Handle chaining if enabled
        if self.config.chaining_enabled:
            self._handle_chaining()
    
    def _update_execution_units(self):
        """Update execution units and complete finished uops"""
        for inst_type in self.execution_units:
            completed_uops = []
            
            for uop_info in self.execution_units[inst_type]:
                uop_id, complete_cycle = uop_info
                if self.current_cycle >= complete_cycle:
                    uop = self.uops[uop_id]
                    uop.completed = True
                    uop.complete_cycle = self.current_cycle
                    completed_uops.append(uop_info)
            
            # Remove completed uops
            for uop_info in completed_uops:
                self.execution_units[inst_type].remove(uop_info)
    
    def _issue_in_order(self):
        """Issue uops in order"""
        for uop in self.uops:
            if not uop.issued and self._can_issue_uop(uop):
                if self._issue_uop(uop):
                    break  # Issue one uop per cycle in strict in-order
    
    def _issue_out_of_order(self):
        """Issue uops out of order within a window"""
        issued_count = 0
        window_size = min(self.config.ooo_window_size, len(self.uops))
        
        # Find the first unissued instruction
        first_unissued = 0
        while (first_unissued < len(self.uops) and 
               self.uops[first_unissued].issued):
            first_unissued += 1
        
        # Look within the window for issuable uops
        window_end = min(first_unissued + window_size, len(self.uops))
        
        for i in range(first_unissued, window_end):
            uop = self.uops[i]
            if not uop.issued and self._can_issue_uop(uop):
                if self._issue_uop(uop):
                    issued_count += 1
                    if issued_count >= 2:  # Multi-issue limit
                        break
    
    def _can_issue_uop(self, uop: MicroOp) -> bool:
        """Check if a uop can be issued"""
        # Check dependencies
        for dep_uop_id in uop.dependencies:
            if not self.uops[dep_uop_id].completed:
                return False
        
        # Check instruction dependencies
        instruction = self.instructions[uop.instruction_id]
        for dep_inst_id in instruction.dependencies:
            if not self.instructions[dep_inst_id].completed:
                return False
        
        return True
    
    def _issue_uop(self, uop: MicroOp) -> bool:
        """Try to issue a uop to an execution unit"""
        # Check resource availability for memory operations
        if uop.type in [InstructionType.LOAD, InstructionType.STORE]:
            # Each memory uop size is typically equal to cache_bandwidth (except possibly the last one)
            # This ensures at most one memory uop can be issued per cycle due to bandwidth constraints
            if self.cache_bandwidth_used + uop.data_size > self.config.cache_bandwidth:
                return False
            self.cache_bandwidth_used += uop.data_size
        
        # Issue the uop
        uop.issued = True
        uop.started = True  
        uop.start_cycle = self.current_cycle
        complete_cycle = self.current_cycle + uop.latency
        
        self.execution_units[uop.type].append((self.uops.index(uop), complete_cycle))
        
        return True
    
    def _handle_chaining(self):
        """Handle chaining between instructions"""
        if not self.config.chaining_enabled:
            return
            
        # Calculate how many elements are ready for each running uop
        for uop in self.uops:
            if uop.started and not uop.completed:
                # Calculate how many elements are ready based on progress
                cycles_elapsed = self.current_cycle - uop.start_cycle
                progress = min(1.0, cycles_elapsed / uop.latency)
                elements_total = uop.data_size // 2  # bf16 elements (2 bytes each)
                uop.ready_elements = int(progress * elements_total)
                
                # For demonstration, print chaining progress
                if cycles_elapsed == 1:  # Only print once per uop
                    producer_inst = self.instructions[uop.instruction_id]
                    if producer_inst.element_wise_dest:
                        print(f"Chaining: uop {uop.instruction_id}.{uop.uop_id} has "
                              f"{uop.ready_elements}/{elements_total} elements ready")
        
        # Note: The actual dependency checking for chaining is handled in _can_issue_uop
        # based on the uop-level dependencies we established in _establish_chaining_dependencies
    
    def _all_instructions_completed(self) -> bool:
        """Check if all instructions have completed"""
        # First update instruction completion status
        for instruction in self.instructions:
            if not instruction.completed:
                uop_ids = self.instruction_uop_map[instruction.id]
                # Set issue_cycle when first uop is issued
                if instruction.issue_cycle == -1:
                    issued_uops = [self.uops[uop_id] for uop_id in uop_ids if self.uops[uop_id].issued]
                    if issued_uops:
                        instruction.issue_cycle = min(uop.start_cycle for uop in issued_uops)
                        instruction.issued = True
                
                if all(self.uops[uop_id].completed for uop_id in uop_ids):
                    instruction.completed = True
                    instruction.complete_cycle = max(self.uops[uop_id].complete_cycle 
                                                   for uop_id in uop_ids)
                    if instruction.start_cycle == -1:
                        instruction.start_cycle = min(self.uops[uop_id].start_cycle 
                                                    for uop_id in uop_ids)
        
        return all(instruction.completed for instruction in self.instructions)
    
    def _generate_results(self) -> Dict:
        """Generate simulation results"""
        # Instruction completion is now handled in _all_instructions_completed()
        
        return {
            'total_cycles': self.current_cycle,
            'instructions': [
                {
                    'id': inst.id,
                    'type': inst.type.value,
                    'issue_cycle': inst.issue_cycle,
                    'start_cycle': inst.start_cycle,
                    'complete_cycle': inst.complete_cycle,
                    'execution_time': inst.complete_cycle - inst.start_cycle if inst.start_cycle >= 0 else -1
                }
                for inst in self.instructions
            ],
            'uops': [
                {
                    'instruction_id': uop.instruction_id,
                    'uop_id': uop.uop_id,
                    'type': uop.type.value,
                    'start_cycle': uop.start_cycle,
                    'complete_cycle': uop.complete_cycle,
                    'execution_time': uop.complete_cycle - uop.start_cycle if uop.start_cycle >= 0 else -1
                }
                for uop in self.uops
            ]
        }
    
    def visualize_execution(self):
        """Generate ASCII visualization of instruction execution timeline"""
        if not self.instructions:
            print("No instructions to visualize")
            return
        
        total_cycles = self.current_cycle
        if total_cycles == 0:
            print("No execution to visualize (0 cycles)")
            return
            
        print("ASCII Execution Timeline:")
        print("@ = Issue, - = Execution, ! = Complete")
        print()
        
        # Print cycle numbers header
        cycle_header = "Instruction".ljust(20) + " " + "".join(f"{i % 10}" for i in range(total_cycles))
        print(cycle_header)
        print("-" * len(cycle_header))
        
        # Generate timeline for each instruction
        for inst in self.instructions:
            timeline = [' '] * total_cycles
            
            # Mark issue cycle with @
            if inst.issue_cycle >= 0 and inst.issue_cycle < total_cycles:
                timeline[inst.issue_cycle] = '@'
            
            # Mark execution cycles with -
            if inst.start_cycle >= 0 and inst.complete_cycle >= 0:
                for cycle in range(max(inst.start_cycle, 0), 
                                 min(inst.complete_cycle, total_cycles)):
                    if timeline[cycle] == ' ':  # Don't overwrite issue marker
                        timeline[cycle] = '-'
            
            # Mark complete cycle with !
            if inst.complete_cycle >= 0 and inst.complete_cycle < total_cycles:
                timeline[inst.complete_cycle] = '!'
            
            # Create instruction label
            inst_label = f"Inst{inst.id} ({inst.type.value})".ljust(20)
            timeline_str = "".join(timeline)
            
            print(f"{inst_label} {timeline_str}")
        
        print()
        print(f"Total execution time: {total_cycles} cycles")
    
    def visualize_uop_execution(self):
        """Generate ASCII visualization of uop execution timeline"""
        if not self.uops:
            print("No uops to visualize")
            return
        
        total_cycles = self.current_cycle
        if total_cycles == 0:
            print("No uop execution to visualize (0 cycles)")
            return
            
        print("ASCII uop Execution Timeline:")
        print("@ = Start, - = Execution, ! = Complete")
        print()
        
        # Print cycle numbers header
        cycle_header = "uop".ljust(25) + " " + "".join(f"{i % 10}" for i in range(total_cycles))
        print(cycle_header)
        print("-" * len(cycle_header))
        
        # Generate timeline for each uop
        for uop in self.uops:
            timeline = [' '] * total_cycles
            
            # Mark start cycle with @
            if uop.start_cycle >= 0 and uop.start_cycle < total_cycles:
                timeline[uop.start_cycle] = '@'
            
            # Mark execution cycles with -
            if uop.start_cycle >= 0 and uop.complete_cycle >= 0:
                for cycle in range(max(uop.start_cycle, 0), 
                                 min(uop.complete_cycle, total_cycles)):
                    if timeline[cycle] == ' ':  # Don't overwrite start marker
                        timeline[cycle] = '-'
            
            # Mark complete cycle with !
            if uop.complete_cycle >= 0 and uop.complete_cycle < total_cycles:
                timeline[uop.complete_cycle] = '!'
            
            # Create uop label with instruction_id.uop_id format
            uop_label = f"uop {uop.instruction_id}.{uop.uop_id} ({uop.type.value})".ljust(25)
            timeline_str = "".join(timeline)
            
            print(f"{uop_label} {timeline_str}")
        
        print()
        print(f"Total uop execution time: {total_cycles} cycles")


def create_softmax_instruction_stream() -> List[Instruction]:
    """Create a sample instruction stream for softmax computation"""
    # Use custom data size (1024 bytes) for this example to maintain compatibility
    # with existing simulation, override the default 256 bytes
    data_size = 2048
    # num_heads = 128
    
    # Softmax typically involves:
    # 1. Load input data
    # 2. Find maximum (reduce)
    # 3. Subtract max from all elements (FMA)
    # 4. Compute exp2 of all elements
    # 5. Sum all exp values (reduce) 
    # 6. Divide by sum (FMA)
    # 7. Store result
    

    # for h in 
    instruction_wrappers = [
        # Load input vector
        LoadInstruction(id=0, target_register=0, data_size=data_size),
        
        # Find maximum value
        ReduceInstruction(id=1, target_register=1, source_registers=[0], 
                         data_size=data_size),
        
        # Subtract max from all elements (x - max)
        FMAInstruction(id=2, target_register=2, source_registers=[1],
                      data_size=data_size),
        
        # Compute exp2(x - max)
        EXP2Instruction(id=3, target_register=3, source_registers=[2],
                       data_size=data_size),
        
        # Sum all exp values
        ReduceInstruction(id=4, target_register=4, source_registers=[3],
                         data_size=data_size),
        
        # Divide by sum (exp / sum)
        FMAInstruction(id=5, target_register=5, source_registers=[3, 4],
                      data_size=data_size),
        
        # Store result
        StoreInstruction(id=6, source_registers=[5], 
                        data_size=data_size)
    ]
    
    # Extract the underlying Instruction objects for compatibility
    return [wrapper.instruction for wrapper in instruction_wrappers]


def main():
    """Example usage of the softmax simulator"""
    # Create processor configuration
    config = ProcessorConfig(
        register_width=2048,
        compute_unit_width=512,
        cache_bandwidth=64,
        execution_mode=ExecutionMode.OUT_OF_ORDER,
        chaining_enabled=True,  # Enable chaining for debugging
        chaining_granularity=64,
        # reduce_latency=4,
        # fma_latency=3,
        # load_latency=2,
        # store_latency=2,
        # exp2_latency=5,
        ooo_window_size=16
    )
    
    print("RISC-V Vector Processor Softmax Simulator")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Register width: {config.register_width} bits")
    print(f"  Compute unit width: {config.compute_unit_width} bits")
    print(f"  Cache bandwidth: {config.cache_bandwidth} bytes/cycle")
    print(f"  Execution mode: {config.execution_mode.value}")
    print(f"  Chaining: {'enabled' if config.chaining_enabled else 'disabled'}")
    if config.chaining_enabled:
        print(f"  Chaining granularity: {config.chaining_granularity} bytes")
    print()
    
    # Create processor
    processor = VectorProcessor(config)
    
    # Load sample softmax instruction stream
    instructions = create_softmax_instruction_stream()
    processor.load_instructions(instructions)
    
    print(f"Loaded {len(instructions)} instructions for softmax computation")
    print("Instructions:")
    for inst in instructions:
        deps_str = f"depends on {list(inst.dependencies)}" if inst.dependencies else "no dependencies"
        print(f"  {inst.id}: {inst.type.value} ({inst.data_size} bytes, {deps_str})")
    print()
    
    # Run simulation with longer timeout now
    results = processor.simulate(max_cycles=1000)
    
    # Print results
    print("Simulation Results:")
    print(f"Total execution time: {results['total_cycles']} cycles")
    print()
    
    print("Instruction Timeline:")
    for inst_result in results['instructions']:
        issue_str = f"issue:{inst_result['issue_cycle']}" if inst_result['issue_cycle'] >= 0 else "issue:N/A"
        print(f"  Instruction {inst_result['id']} ({inst_result['type']}): "
              f"{issue_str}, cycles {inst_result['start_cycle']}-{inst_result['complete_cycle']} "
              f"(duration: {inst_result['execution_time']})")
    print()
    
    print(f"Generated {len(results['uops'])} micro-operations")
    print()
    
    # Print detailed uop timeline
    print("Micro-operation (uop) Timeline:")
    for uop_result in results['uops']:
        inst_id = uop_result['instruction_id']
        uop_id = uop_result['uop_id']
        uop_type = uop_result['type']
        start_cycle = uop_result['start_cycle']
        complete_cycle = uop_result['complete_cycle']
        execution_time = uop_result['execution_time']
        
        # Use consistent format with instruction output, adding uop_id annotation
        print(f"  uop {inst_id}.{uop_id} ({uop_type}): "
              f"cycles {start_cycle}-{complete_cycle} "
              f"(duration: {execution_time})")
    print()
    
    # Display ASCII visualization
    processor.visualize_execution()
    print()
    
    # Display uop-level ASCII visualization  
    processor.visualize_uop_execution()
    print()
    
    # Calculate some performance metrics
    if results['instructions']:
        throughput = len(instructions) / results['total_cycles']
        print(f"Instruction throughput: {throughput:.3f} instructions/cycle")


if __name__ == "__main__":
    main()
