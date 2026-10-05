## Concepts: 

Xid 79: "GPU has fallen off the bus" => NVIDIA driver has lost communication with the GPU over the PCIe (Peripheral Component Interconnect Express) bus. OS cannot detect or interact with GPU.
Check: dmesg -T | grep -i xid
Causes: Physical connections (reseat GPU, check power cables), H/w failure, overheating (ex: in dg05, GPU0's power was capped @ 200W to avoid xid79 due to overheating) 




Issue: 
### nvidia-smi is showing Persistence-M as OFF because the nvidia-persistenced service has "--no-persistence-mode" in ExecStart.
Fix: nvidia-smi -pm 1


hostedai@dg05:~/learning-lab$ nvidia-smi
Mon Oct  5 06:16:32 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.91.07              Driver Version: 595.91.07      CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100 80GB PCIe          Off |   00000000:01:00.0 Off |                    0 |
| N/A   42C    P0             47W /  200W |       0MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA A100 80GB PCIe          Off |   00000000:42:00.0 Off |                    0 |
| N/A   37C    P0             44W /  300W |       0MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+


hostedai@dg05:~/learning-lab$ sudo systemctl status nvidia-persistenced --no-pager -l
● nvidia-persistenced.service - NVIDIA Persistence Daemon
     Loaded: loaded (/usr/lib/systemd/system/nvidia-persistenced.service; static)
     Active: active (running) since Mon 2026-10-05 03:58:47 UTC; 2h 18min ago
   Main PID: 1934 (nvidia-persiste)
      Tasks: 1 (limit: 618589)
     Memory: 600.0K (peak: 1.0M)
        CPU: 4ms
     CGroup: /system.slice/nvidia-persistenced.service
             └─1934 /usr/bin/nvidia-persistenced --user nvidia-persistenced --no-persistence-mode --verbose


NVIDIA GPU Persistence Mode
- prevents the Linux kernel from unloading the NVIDIA driver when no applications are actively using the GPU. 
- eliminates driver initialization latency for compute/headless workloads and ensures custom configurations (e.g., power limits, clock offsets) are maintained across idle periods.

Mechanisms of Persistence
NVIDIA supports two distinct methods for achieving state persistence:
	1.	Legacy Kernel Persistence: Managed via nvidia-smi -pm 1. This instructs the kernel-space driver to stay awake. The Persistence-M column in nvidia-smi strictly monitors this legacy kernel state.
	2.	User-Space Daemon (nvidia-persistenced): The modern, resource-efficient approach. It runs a lightweight background daemon that holds the GPU character device files open. By maintaining an open file descriptor, it prevents the Linux kernel from tearing down the driver state.

The nvidia-smi Discrepancy
By default on many Debian/Ubuntu-based distributions, the systemd service for the persistence daemon is configured with the --no-persistence-mode flag:
ExecStart=/usr/bin/nvidia-persistenced --user nvidia-persistenced --no-persistence-mode
⚬	Impact: The daemon successfully keeps the driver loaded via file descriptors (achieving the actual goal of persistence), but explicitly refrains from toggling the legacy kernel switch.
⚬	Result: nvidia-smi will display Persistence-M: Off, leading to a false negative in monitoring tools, despite the GPUs actively persisting in the background.
Resolution
To align the daemon's behavior with the legacy nvidia-smi reporting output, the flag must be removed.
	1.	Modify the systemd service file:
sudo nano /usr/lib/systemd/system/nvidia-persistenced.service

	2.	Update the ExecStart parameter by removing --no-persistence-mode:
ExecStart=/usr/bin/nvidia-persistenced --user nvidia-persistenced --verbose

	3.	Apply changes and restart the daemon:
sudo systemctl daemon-reload
sudo systemctl restart nvidia-persistenced

	4.	Validation:
Executing nvidia-smi will now reflect Persistence-M: On for all initialized GPUs.

Applying Persistent Configurations

Nvidia-persistenced only keeps the driver loaded.

To ensure settings like power caps (e.g., restricting an A100 to 200W) persist across reboots, administrators should utilize systemd to apply nvidia-smi configurations at startup:

Ex: Tuning parameters to cap power and clocks are added into below systemd config file to persist across reboots.

root@dg05:~# systemctl cat haishare-gpu0-thermal.service
# /etc/systemd/system/haishare-gpu0-thermal.service
[Unit]
Description=HAISHARE GPU0 thermal mitigation (power cap + clock lock) - Xid 79 workaround
Documentation=https://github.com/hosted-ai/haishare-multigpu (a100-validation)
After=nvidia-persistenced.service multi-user.target
Wants=nvidia-persistenced.service

[Service]
Type=oneshot
RemainAfterExit=yes
# GPU0 (PCI 01:00) runs ~30C hotter than peers and throws Xid 79 under load.
# Cap power + clocks to reduce peak heat. || true so a missing/healthy GPU0
# never blocks boot. Remove this unit once GPU0 cooling is serviced.
ExecStart=/bin/sh -c "/usr/bin/nvidia-smi -i 0 -pl 200 || true" 
ExecStart=/bin/sh -c "/usr/bin/nvidia-smi -i 0 -lgc 210,1200 || true"

[Install]
WantedBy=multi-user.target
root@dg05:~#

	
Note:    
-i,   --id=                 Target a specific GPU or Unit.    
-lgc  --lock-gpu-clocks=    Specifies <minGpuClock,maxGpuClock> clocks as a pair (e.g. 1500,1500) that defines the range of desired locked GPU clock speed in MHz. Setting this will supersede application clocks and take effect regardless if an app is running.


System Resource Implications
⚬	Memory Footprint: user-space nvidia-persistenced daemon = lightweight (consumes <2MB RAM) => better than keeping a dummy CUDA process alive.
⚬	Idle Power Consumption: While the driver remains loaded, the GPU will correctly drop into its lowest idle power state (e.g., P8 state) unless explicitly overridden by Prefer Maximum Performance settings. Persistence mode does not force the GPU to draw maximum power while idle.

Verification & Diagnostics
Beyond checking the Persistence-M flag, you can verify the daemon's active file descriptor holds using the lsof command:
sudo lsof /dev/nvidia*

If the daemon is functioning correctly, you will see nvidia-persistenced listed as actively holding open /dev/nvidia0, /dev/nvidia1, and /dev/nvidiactl. This confirms the user-space lock on the driver state is active.



However, in dg05, lsof /dev/nvidia* output was empty. To fix, removed "--no-persistence-mode" from /usr/lib/systemd/system/nvidia-persistenced.service and restarted the service.
Post this nvidia-smi was showing Persistence-M as "On".

root@dg05:~# sudo sed -i 's/--no-persistence-mode //g' /usr/lib/systemd/system/nvidia-persistenced.service
root@dg05:~# sudo systemctl daemon-reload
root@dg05:~# sudo systemctl restart nvidia-persistenced
root@dg05:~# sudo systemctl statsu nvidia-persistenced
Unknown command verb 'statsu', did you mean 'status'?
root@dg05:~# sudo systemctl status nvidia-persistenced
● nvidia-persistenced.service - NVIDIA Persistence Daemon
     Loaded: loaded (/usr/lib/systemd/system/nvidia-persistenced.service; static)
     Active: active (running) since Mon 2026-10-05 17:18:15 UTC; 11s ago
    Process: 56683 ExecStart=/usr/bin/nvidia-persistenced --user nvidia-persistenced --verbose (code=>
   Main PID: 56684 (nvidia-persiste)
      Tasks: 1 (limit: 618589)
     Memory: 308.0K (peak: 1.7M)
        CPU: 29ms
     CGroup: /system.slice/nvidia-persistenced.service
             └─56684 /usr/bin/nvidia-persistenced --user nvidia-persistenced --verbose

Oct 05 17:18:15 dg05 nvidia-persistenced[56684]: device 0000:42:00.0 - persistence mode enabled.
Oct 05 17:18:15 dg05 nvidia-persistenced[56684]: device 0000:42:00.0 - NUMA memory onlined.
Oct 05 17:18:15 dg05 nvidia-persistenced[56684]: device 0000:81:00.0 - registered
Oct 05 17:18:15 dg05 nvidia-persistenced[56684]: device 0000:81:00.0 - persistence mode enabled.
Oct 05 17:18:15 dg05 nvidia-persistenced[56684]: device 0000:81:00.0 - NUMA memory onlined.
Oct 05 17:18:15 dg05 nvidia-persistenced[56684]: device 0000:c1:00.0 - registered
Oct 05 17:18:15 dg05 nvidia-persistenced[56684]: device 0000:c1:00.0 - persistence mode enabled.
Oct 05 17:18:15 dg05 nvidia-persistenced[56684]: device 0000:c1:00.0 - NUMA memory onlined.
Oct 05 17:18:15 dg05 nvidia-persistenced[56684]: Local RPC services initialized
Oct 05 17:18:15 dg05 systemd[1]: Started nvidia-persistenced.service - NVIDIA Persistence Daemon.
root@dg05:~# nvidia-smi
Mon Oct  5 17:18:42 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.91.07              Driver Version: 595.91.07      CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100 80GB PCIe          On  |   00000000:01:00.0 Off |                    0 |
| N/A   34C    P0             45W /  200W |       0MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA A100 80GB PCIe          On  |   00000000:42:00.0 Off |                    0 |
| N/A   31C    P0             42W /  300W |       0MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+





Clock locking:

GPU's core clock (the SM clock) isn't fixed. It boosts up when there's headroom and drops when the card hits its power limit or gets hot. So the same benchmark can run at 1,410 MHz one minute and 1,250 MHz the next, and your numbers move with it.

"$ nvidia-smi -i 2 -q -d SUPPORTED_CLOCKS" lists the allowed values. You'll see one memory clock (1512 MHz, which on the A100 is fixed, so you can't lock it) and a list of SM clocks from about 210 MHz up to 1410 MHz. Locking means telling the GPU to stay at one SM value: sudo nvidia-smi -i 2 -lgc 1410,1410.

The choice: 1410 is the maximum. It gives the best performance, but under heavy load the card may hit its 300W cap and drop below it anyway, so you aren't really locked. A lower value (say 1200) holds steady but throws away speed.

How to decide: start with 1410. During your first benchmark, watch nvidia-smi -i 2 -q -d CLOCK,PERFORMANCE and look at the actual SM clock and the throttle reasons. If it holds 1410, keep it. If it dips, lower the lock until it stays put.