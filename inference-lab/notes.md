## Concepts: 

Xid 79: "GPU has fallen off the bus" => NVIDIA driver has lost communication with the GPU over the PCIe (Peripheral Component Interconnect Express) bus. OS cannot detect or interact with GPU.
Check: dmesg -T | grep -i xid
Causes: Physical connections (reseat GPU, check power cables), H/w failure, overheating (ex: in dg05, GPU0's power was capped @ 200W to avoid xid79 due to overheating) 



NVIDIA GPU Persistence Mode
- Prevents OS kernel from unloading NVIDIA driver even when no processes are actively using the GPUs.
- Why needed : It eliminates driver initialization latency for compute/headless workloads.

Note:
- compute workload: mathematical operations that use GPU only for parallel processing. 
- headless: server without attached monitor or GUI. managed via ssh and do not have display manager like X11

NVIDIA supports two methods for achieving state persistence:

	1.	Legacy Kernel Persistence: 
      Managed via nvidia-smi -pm 1. This instructs the kernel-space driver to stay awake. The Persistence-M column in nvidia-smi monitors this legacy kernel state.

	2.	User-Space Daemon (nvidia-persistenced): 
      It holds the GPU character device files open. By maintaining an open file descriptor, it prevents the Linux kernel from unloading the driver state.
      In ubuntu, nvidia-persistenced is configured with --no-persistence-mode flag: ExecStart=/usr/bin/nvidia-persistenced --user nvidia-persistenced --no-persistence-mode

      Ex:
      root@dg05:~# sudo systemctl edit nvidia-persistenced
      Successfully installed edited file '/etc/systemd/system/nvidia-persistenced.service.d/override.conf'.
      
      root@dg05:~# cat /etc/systemd/system/nvidia-persistenced.service.d/override.conf
      ExecStart=/usr/bin/nvidia-persistenced --user nvidia-persistenced --verbose
      
      root@dg05:~# sudo systemctl daemon-reload ; sudo systemctl restart nvidia-persistenced

	   Validation: 
      1. $ nvidia-smi => "Persistence-M: On" for all GPUs.
      2. lsof /dev/nvidia* => must list the open GPU character device files (/dev/nvidiactl, /dev/nvidia0, /dev/nvidia1)

         Ex: root@dg05:~# lsof /dev/nvidia*
               COMMAND     PID                USER   FD   TYPE  DEVICE SIZE/OFF NODE NAME
               nvidia-pe 56684 nvidia-persistenced    2u   CHR 195,255      0t0  997 /dev/nvidiactl
               nvidia-pe 56684 nvidia-persistenced    3u   CHR   195,0      0t0 1001 /dev/nvidia0
               nvidia-pe 56684 nvidia-persistenced    5u   CHR   195,0      0t0 1001 /dev/nvidia0

 Tuning Power and GPU Clocks: Need to create a separate systemd service to make these persistent.
 
Ex: To set max power and clocks refer ExecStart in below systemd config file. This persists across reboots.

root@dg05:~# systemctl cat haishare-gpu0-thermal.service
# /etc/systemd/system/haishare-gpu0-thermal.service
[Unit]
Description=HAISHARE GPU0 thermal mitigation (power cap + clock lock) - Xid 79 workaround
After=nvidia-persistenced.service multi-user.target
Wants=nvidia-persistenced.service

[Service]
Type=oneshot
RemainAfterExit=yes
# GPU0 (PCI 01:00) runs ~30C hotter than peers and throws Xid 79 under load.
# Cap power + clocks to reduce peak heat. || true so a missing/healthy GPU0
# never blocks boot. Remove this unit once GPU0 cooling is serviced.
ExecStart=/bin/sh -c "/usr/bin/nvidia-smi -i 0 -pl 200 || true"   ## set Max power to 200W for GPU0.
ExecStart=/bin/sh -c "/usr/bin/nvidia-smi -i 0 -lgc 210,1200 || true" ## sets minGpuClock,maxGpuClock for GPU0

[Install]
WantedBy=multi-user.target

root@dg05:~#
root@dg05:~# sudo systemctl daemon-reload
root@dg05:~# sudo systemctl restart nvidia-persistenced


Note:    
-i,   --id=                 Target a specific GPU or Unit.    
-lgc  --lock-gpu-clocks=    Specifies <minGpuClock,maxGpuClock> clocks as a pair (e.g. 1500,1500) => range of desired locked GPU clock speed in MHz.

## Reference

GPU Clock locking:
GPU's core clock (the SM clock) isn't fixed. It increases with workload and drops when the GPU reaches its power limit or gets hot. => same benchmark can run at 1,410 MHz one minute and 1,250 MHz the next, and your numbers move with it.

$ nvidia-smi -i 2 -q -d SUPPORTED_CLOCKS        ### lists the allowed values. For A100, SM clocks range from 210 MHz to 1410 MHz. 

Locking => telling the GPU to stay at one SM value: sudo nvidia-smi -i 2 -lgc 1410,1410.

Note: 1512 MHz memory clock on the A100 is fixed and cant be locked. 

Tradeoff: For A100, 1410 MHz is maximum. GPU under high load can reach 300W and hence drops GPU clock below 1410 MHz even when locked. A lower value (say 1200) holds steady but at reduced speed.

How to decide: start with 1410. During your first benchmark, watch nvidia-smi -i 2 -q -d CLOCK,PERFORMANCE and look at the actual SM clock and the throttle reasons. If it holds 1410, keep it. If it dips, lower the lock until it stays put.