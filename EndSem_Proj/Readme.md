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
We have implemented this by keeping track of the number of instructions issued by each CTA in a shader core. We then, have used this data to sort the `m_next_cycle_prioritized_warps` after the max CTAs have been issued for that core.
Here's the code for the same:
```cpp
void scheduler_unit::order_warps_max_cta_issued(){
  std::vector<shd_warp_t*> temp(m_supervised_warps);
  std::sort(temp.begin(), temp.end(),
            [this](shd_warp_t* lhs, shd_warp_t* rhs) {
              int st1 = m_shader->m_cta_status[lhs->get_cta_id()];
              int st2 = m_shader->m_cta_status[rhs->get_cta_id()];
              int inst1 = m_shader->cta_inst_issued[ lhs->get_cta_id() ];
              int inst2 = m_shader->cta_inst_issued[ rhs->get_cta_id() ];
        return compare_cta_inst(st1, st2, inst1, inst2);
    }
  );

  m_next_cycle_prioritized_warps.clear();
  m_next_cycle_prioritized_warps = temp;
}
```

### 2. Warp Sharing
We have implemented warp sharing by modifying the `m_supervised_warps` of each scheduler before calling the `cycle` function on them. This way they have the optimal set of warps assigned to them before each cycle for preventing stalls and ensuring resource utilization.
The following functions implement the mechanism:

#### 1. get_shared_warps
The `get_shared_warps` function is part of a scheduler unit. This function is responsible for determining the availability and categorization of supervised warps based on their instruction types. It takes as input an array representing the availability of different operation units (`availability[]`) and three vectors (`sp_inst`, `sfu_inst`, and `mem_inst`) to store pairs of scheduler ID and corresponding supervised warp. The function also receives a scheduler ID (`sched_id`).

For each supervised warp in the scheduler unit's list (`m_supervised_warps`), the function checks if its instruction buffer is not empty. If the buffer is not empty, it retrieves the next instruction and determines its type using the `get_inst_type` function. Depending on the instruction type, it checks the availability of the corresponding operation unit and updates the availability mask accordingly. If the operation unit is available, the supervised warp is added to the respective vector along with the scheduler ID. Finally, the function updates the availability array for the specific scheduler with the modified availability mask.
```cpp
void scheduler_unit::get_shared_warps(unsigned int availability[], std::vector<std::pair<unsigned int, shd_warp_t*>> &sp_inst, std::vector<std::pair<unsigned int, shd_warp_t*>> &sfu_inst, std::vector<std::pair<unsigned int, shd_warp_t*>> &mem_inst, unsigned int sched_id){
  
  unsigned int ocu_avail_mask = get_ocu_availability_mask(), temp_ocu_avail_mask = ocu_avail_mask;
  
  for(int i=0; i<m_supervised_warps.size(); i++){
    shd_warp_t* w = m_supervised_warps[i];
    if(w->ibuffer_empty()) continue;
    const warp_inst_t* inst = warp(w->get_warp_id()).ibuffer_next_inst();
    unsigned int inst_type = get_inst_type(inst);
    if(inst_type!=-1){
      if(inst_type==0){
        if(!(temp_ocu_avail_mask & (1<<0)))
          sp_inst.push_back(std::make_pair(sched_id, w));
        else
          ocu_avail_mask &= ~(1<<0);
      }
      else if(inst_type==1){
        if(!(temp_ocu_avail_mask & (1<<1)))
          sfu_inst.push_back(std::make_pair(sched_id, w));
        else
          ocu_avail_mask &= ~(1<<1);
      }
      else if(inst_type==2){
        if(!(temp_ocu_avail_mask & (1<<2)))
          mem_inst.push_back(std::make_pair(sched_id, w));
        else
          ocu_avail_mask &= ~(1<<2);
      }
    }
  }

  availability[sched_id] = ocu_avail_mask;
}
```
#### 2. get_ocu_availability_mask
The `get_ocu_availability_mask` function calculates and returns a bit mask representing the availability of different OCUs (operand collector units) in the scheduler unit. Each bit in the mask corresponds to a specific OCU (SP, SFU, and Memory). The function checks if each unit has available slots and sets the corresponding bit in the mask accordingly.
The rightmost bit is for SP, middle for SFU, and leftmost for MEM.
```cpp
unsigned int scheduler_unit::get_ocu_availability_mask(){
  unsigned int fu_avail_mask = 0;
  fu_avail_mask |= (m_sp_out->has_free(m_shader->m_config->sub_core_model, m_id) << 0);
  fu_avail_mask |= (m_sfu_out->has_free(m_shader->m_config->sub_core_model, m_id) << 1);
  fu_avail_mask |= (m_mem_out->has_free(m_shader->m_config->sub_core_model, m_id) << 2);
  return fu_avail_mask;
}
```

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
```cpp
for(int i=0;i<schedulers.size();i++){
  unsigned int ocu_availabily = availability[i];
      if(ocu_availabily & (1<<0)){
        if(sp_inst_warps.size()>0){
          std::vector<std::pair<unsigned int, shd_warp_t*>> :: iterator it = sp_inst_warps.begin();
          // Adding warp to scheduler with available OCU
          schedulers[i]->add_shared_warp((*it).second->get_warp_id());
          // Removing warp from original scheduler
          schedulers[(*it).first]->remove_shared_warp((*it).second);
          // Erasing warp from vector
          sp_inst_warps.erase(it);\
          // Done sharing warp
        }
      }
      if(ocu_availabily & (1<<1)){
        if(sfu_inst_warps.size()>0){
          std::vector<std::pair<unsigned int, shd_warp_t*>> :: iterator it = sfu_inst_warps.begin();
          schedulers[i]->add_shared_warp((*it).second->get_warp_id());
          schedulers[(*it).first]->remove_shared_warp((*it).second);
          sfu_inst_warps.erase(it);
        }
      }
      if(ocu_availabily & (1<<2)){
        if(mem_inst_warps.size()>0){
          std::vector<std::pair<unsigned int, shd_warp_t*>> :: iterator it = mem_inst_warps.begin();
          schedulers[i]->add_shared_warp((*it).second->get_warp_id());
          schedulers[(*it).first]->remove_shared_warp((*it).second);
          mem_inst_warps.erase(it);
        }
      }
}
```

## Benchmark Analysis
![Alt text](./KAWS%20(1).png?raw=true "Figure 1")
![Alt text](./2.11%25.png?raw=true "Figure 2")
![Alt text](./2.11%25%20(1).png?raw=true "Figure 3")