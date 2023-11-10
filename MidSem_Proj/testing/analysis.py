issued = 0
xalu = 0
xmem = 0
waiting = 0
others = 0

warp_null = 0
warp_done = 0
cdp1 = 0
cdp2 = 0
control_hazard = 0
diverge_return = 0

not_checked = 0

_issued = 0
_xalu = 0
_xmem = 0
_waiting = 0
_others = 0

_warp_null = 0
_warp_done = 0
_cdp1 = 0
_cdp2 = 0
_control_hazard = 0
_diverge_return = 0


kernel_issued = 0
kernel_xalu = 0
kernel_xmem = 0
kernel_waiting = 0
kernel_others = 0

kernel_warp_null = 0
kernel_warp_done = 0
kernel_cdp1 = 0
kernel_cdp2 = 0
kernel_control_hazard = 0
kernel_diverge_return = 0

kernel_not_checked = 0

kernel__issued = 0
kernel__xalu = 0
kernel__xmem = 0
kernel__waiting = 0
kernel__others = 0

kernel__warp_null = 0
kernel__warp_done = 0
kernel__cdp1 = 0
kernel__cdp2 = 0
kernel__control_hazard = 0
kernel__diverge_return = 0


sim_cycles = -1
num_warps = -1
called = 0
pipeline_cycles = 0
num_kernels = 0
prev_cycles = 0


idx = 0

with open("out.txt", "r") as f:
    lines = f.readlines()

for line in lines:

    if("!@#$" in line):
        if(line == "!@#$ISSUED\n"):
            kernel_issued += 1
        elif(line == "!@#$WAITING\n"):
            kernel_waiting += 1
        elif(line == "!@#$OTHER\n"):
            kernel_others += 1
        elif(line == "!@#$XALU\n"):
            kernel_xalu += 1
        elif(line == "!@#$XMEM\n"):
            kernel_xmem += 1
        elif("!@#$WARPS_NOT_CHECKED" in line):
            kernel_not_checked += int(line.split()[-1])
        elif(line == "!@#$CALLED\n"):
            called += 1
        elif(line == "!@#$PIPELINE_CYCLE\n"):
            pipeline_cycles += 1
        elif(line == "!@#$NULL\n"):
            kernel_warp_null += 1
        elif(line == "!@#$DONE_EXIT\n"):
            kernel_warp_done += 1
        elif(line == "!@#$CDP1\n"):
            kernel_cdp1 += 1
        elif(line == "!@#$CDP2\n"):
            kernel_cdp2 += 1
        elif(line == "!@#$CONTROL_HAZARD\n"):
            kernel_control_hazard += 1
        elif(line == "!@#$DIVERGE_RETURN\n"):
            kernel_diverge_return += 1
        elif(line == "!@#$_ISSUED\n"):
            kernel__issued += 1
        elif(line == "!@#$_WAITING\n"):
            kernel__waiting += 1
        elif(line == "!@#$_OTHER\n"):
            kernel__others += 1
        elif(line == "!@#$_XALU\n"):
            kernel__xalu += 1
        elif(line == "!@#$_XMEM\n"):
            kernel__xmem += 1
        elif(line == "!@#$_NULL\n"):
            kernel__warp_null += 1
        elif(line == "!@#$_DONE_EXIT\n"):
            kernel__warp_done += 1
        elif(line == "!@#$_CDP1\n"):
            kernel__cdp1 += 1
        elif(line == "!@#$_CDP2\n"):
            kernel__cdp2 += 1
        elif(line == "!@#$_CONTROL_HAZARD\n"):
            kernel__control_hazard += 1
        elif(line == "!@#$_DIVERGE_RETURN\n"):
            kernel__diverge_return += 1
    elif("kernel_launch_uid = " in line):
        print(line)
        num_kernels += 1
        #add all kernel_* variables to corresponding variables and assign them to zero
        issued += kernel_issued
        xalu += kernel_xalu
        xmem += kernel_xmem
        waiting += kernel_waiting
        others += kernel_others

        warp_null += kernel_warp_null
        warp_done += kernel_warp_done
        cdp1 += kernel_cdp1
        cdp2 += kernel_cdp2
        control_hazard += kernel_control_hazard
        diverge_return += kernel_diverge_return

        not_checked += kernel_not_checked

        _issued += kernel__issued
        _xalu += kernel__xalu
        _xmem += kernel__xmem
        _waiting += kernel__waiting
        _others += kernel__others

        _warp_null += kernel__warp_null
        _warp_done += kernel__warp_done
        _cdp1 += kernel__cdp1
        _cdp2 += kernel__cdp2
        _control_hazard += kernel__control_hazard
        _diverge_return += kernel__diverge_return

        print("Issued: ", kernel_issued)
        print("Waiting: ", kernel_waiting)
        print("XALU: ", kernel_xalu)
        print("XMEM: ", kernel_xmem)
        print("Others: ", kernel_others)

        print("Warp Null: ", kernel_warp_null)
        print("Warp Done: ", kernel_warp_done)
        print("CDP1: ", kernel_cdp1)
        print("CDP2: ", kernel_cdp2)
        print("Control Hazard: ", kernel_control_hazard)
        print("Diverge Return: ", kernel_diverge_return)

        print("Not Checked: ", kernel_not_checked)

        print("_Issued: ", kernel__issued)
        print("_Waiting: ", kernel__waiting)
        print("_XALU: ", kernel__xalu)
        print("_XMEM: ", kernel__xmem)
        print("_Others: ", kernel__others)

        print("_Warp Null: ", kernel__warp_null)
        print("_Warp Done: ", kernel__warp_done)
        print("_CDP1: ", kernel__cdp1)
        print("_CDP2: ", kernel__cdp2)
        print("_Control Hazard: ", kernel__control_hazard)
        print("_Diverge Return: ", kernel__diverge_return)

        print("\nTotal Entries", kernel_issued+kernel_waiting+kernel_xalu+kernel_xmem+kernel_others
                                +kernel_warp_null+kernel_warp_done+kernel_cdp1+kernel_cdp2+kernel_control_hazard+kernel_diverge_return
                                +kernel__issued+kernel__waiting+kernel__xalu+kernel__xmem+kernel__others
                                +kernel__warp_null+kernel__warp_done+kernel__cdp1+kernel__cdp2+kernel__control_hazard+kernel__diverge_return)

        kernel_issued = 0
        kernel_xalu = 0
        kernel_xmem = 0
        kernel_waiting = 0
        kernel_others = 0

        kernel_warp_null = 0
        kernel_warp_done = 0
        kernel_cdp1 = 0
        kernel_cdp2 = 0
        kernel_control_hazard = 0
        kernel_diverge_return = 0

        kernel_not_checked = 0

        kernel__issued = 0
        kernel__xalu = 0
        kernel__xmem = 0
        kernel__waiting = 0
        kernel__others = 0

        kernel__warp_null = 0
        kernel__warp_done = 0
        kernel__cdp1 = 0
        kernel__cdp2 = 0
        kernel__control_hazard = 0
        kernel__diverge_return = 0
        
    elif("gpu_tot_sim_cycle" in line):
        sim_cycles = int(line.split()[-1])
        print("Total Expected Entries = ", 24*(sim_cycles-prev_cycles-1)*2)
        print(line)
        prev_cycles = sim_cycles
    elif("warp_id:" in line):
        # print(lines[idx+1])
        num_warps = len(lines[idx+1].split())

    idx += 1

print("----------------------------------------")
print("\nNum Kernels: ", num_kernels)
print("\nTotal Sim Cycles: ", sim_cycles)
print("Num Warps: ", num_warps)

print("\nIssued: ", issued)
print("Waiting: ", waiting)
print("XALU: ", xalu)
print("XMEM: ", xmem)
print("Others: ", others)

print("\nWarp Null: ", warp_null)
print("Warp Done: ", warp_done)
print("CDP1: ", cdp1)
print("CDP2: ", cdp2)
print("Control Hazard: ", control_hazard)
print("Diverge Return: ", diverge_return)


print("\n_Issued: ", _issued)
print("_Waiting: ", _waiting)
print("_XALU: ", _xalu)
print("_XMEM: ", _xmem)
print("_Others: ", _others)

print("\n_Warp Null: ", _warp_null)
print("_Warp Done: ", _warp_done)
print("_CDP1: ", _cdp1)
print("_CDP2: ", _cdp2)
print("_Control Hazard: ", _control_hazard)
print("_Diverge Return: ", _diverge_return)

print("\nNot including warps not checked\n")
print("Actual Entries", issued+waiting+xalu+xmem+others)
print("Actual Entries(including control hazard, etc)", issued+waiting+xalu+xmem+others+warp_null+warp_done+cdp1+cdp2+control_hazard+diverge_return)

print("\nIncluding warps not checked")
print("Warps not Checked: ", not_checked)
print("\nActual Entries", issued+waiting+xalu+xmem+others+_issued+_waiting+_xalu+_xmem+_others)
print("Actual Entries(including control hazard, etc)", issued+waiting+xalu+xmem+others
                                                    +_issued+_waiting+_xalu+_xmem+_others 
                                                    + warp_null+warp_done+cdp1+cdp2+control_hazard+diverge_return
                                                    +_warp_null+_warp_done+_cdp1+_cdp2+_control_hazard+_diverge_return)
print("\nExpected total entries", 24*(sim_cycles-num_kernels)*2)
print("\nCalled: ", called)
print("\nPipeline Cycles: ", pipeline_cycles)