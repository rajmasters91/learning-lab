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

