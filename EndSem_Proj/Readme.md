# Implementation of KAWS: Coordinate Kernel-Aware Warp Scheduling and Warp Sharing Mechanism for Advanced GPUs

### Usage
Add the following parameters to the gpgpusim.config file.
```
-gpgpu_enable_kernel_aware_warp_scheduling 1
-gpgpu_enable_warp_sharing 1
```
Set the value to 1 for enabling and 0 for disabling the feature.

## Implementation

### 1. Kernel-Aware Warp Scheduler
We have implemented this by keeping track of the number of instructions issued by each CTA in a shader core. We then, have used this data to sort the `m_next_cycle_prioritized_warps` after the max CTAs have been issued for that core in the function `order_warps_max_cta_issued` which is a part of the scheduler_unit class.

### 2. Warp Sharing
We have implemented warp sharing by modifying the `m_supervised_warps` of each scheduler before calling the `cycle` function on them. This way they have the optimal set of warps assigned to them before each cycle for preventing stalls and ensuring resource utilization.
The following functions implement the mechanism:

#### 1. get_shared_warps
The `get_shared_warps` function is part of a scheduler unit. This function is responsible for determining the availability and categorization of supervised warps based on their instruction types. It takes as input an array representing the availability of different operation units (`availability[]`) and three vectors (`sp_inst`, `sfu_inst`, and `mem_inst`) to store pairs of scheduler ID and corresponding supervised warp. The function also receives a scheduler ID (`sched_id`).

For each supervised warp in the scheduler unit's list (`m_supervised_warps`), the function checks if its instruction buffer is not empty. If the buffer is not empty, it retrieves the next instruction and determines its type using the `get_inst_type` function. Depending on the instruction type, it checks the availability of the corresponding operation unit and updates the availability mask accordingly. If the operation unit is available, the supervised warp is added to the respective vector along with the scheduler ID. Finally, the function updates the availability array for the specific scheduler with the modified availability mask.

#### 2. get_ocu_availability_mask
The `get_ocu_availability_mask` function calculates and returns a bit mask representing the availability of different OCUs (operand collector units) in the scheduler unit. Each bit in the mask corresponds to a specific OCU (SP, SFU, and Memory). The function checks if each unit has available slots and sets the corresponding bit in the mask accordingly.
The rightmost bit is for SP, middle for SFU, and leftmost for MEM.

#### 3. get_inst_type
The `get_inst_type` function determines the type of instruction (operation) represented by a given warp instruction (`warp_inst_t`). It categorizes instructions into three types: 

- Type 0 (`return 0`): Corresponds to instructions related to scalar ALU operations, where the operation involves integer arithmetic or logic (SP operation).
- Type 1 (`return 1`): Corresponds to instructions related to special function units (SFU), double-precision operations, or ALU-SFU combinations.
- Type 2 (`return 2`): Corresponds to memory-related instructions, including load, store, memory barrier, and tensor core operations.

#### 4. Sharing warps with appropriate schedulers
This loop is a part of the `shader_core_ctx::issue()` function. It distributes shared warps among different schedulers based on the availability of Operational Compute Units (OCUs) for each scheduler. Here's a breakdown of what it does:

1. **Iteration through Schedulers:** The loop iterates through each scheduler in the `schedulers` vector.

2. **Availability Check:** For each scheduler, it checks the availability of OCUs (`ocu_availabily`) using a bit mask. The bit at position 0 corresponds to the availability of OCU 0, the bit at position 1 corresponds to OCU 1, and so on.

3. **Sharing Warps:** For each available OCU, it checks if there are warps waiting to be scheduled for that OCU type (e.g., SP, SFU, Memory). If warps are available, it selects the first warp in the corresponding vector (`sp_inst_warps`, `sfu_inst_warps`, `mem_inst_warps`).

4. **Scheduler Update:** It then adds the selected warp to the current scheduler using `schedulers[i]->add_shared_warp(...)`, removes the warp from its original scheduler (`schedulers[(*it).first]->remove_shared_warp(...)`), and erases it from the vector of warps waiting to be scheduled.

5. **Iteration Continues:** The loop repeats this process for each type of OCU, ensuring that available warps are appropriately distributed among the schedulers based on OCU availability.

The overall goal is to efficiently distribute the available warps among the schedulers, taking into account the availability of different types of OCUs in each scheduler.

## Benchmark Analysis

### Data
<p align="center">
  <img src="./KAWS%20(1).png?raw=true" /><br/>
</p>

### IPC plot without warp sharing
<p align="center">
  <img src="./2.11%25%20(1).png?raw=true" /><br/>
</p>

#### Improvement
Average : 2.94%    Max : 6.23%

### IPC plot with warp sharing
<p align="center">
  <img src="./2.11%25.png?raw=true" /><br/>
</p>

#### Improvement
Average : 3.38%    Max : 7.32%
