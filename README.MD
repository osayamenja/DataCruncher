Data Cruncher
------
## Nsight Traces
We open source our traces in Azure Blob storage. 
You can generate all datasets available in [data](./data)
by downloading the linked files and running their corresponding commands. 

## Note
The time filtering in the commands are to restrict the data to complete iterations. 
You can verify this claim by viewing the traces in the GUI.

## Requirements
- CUDA Toolkit
- Linux
- Python
- Nsight Systems CLI and GUI

## Single-Node 1x8 350M
- Download Trace from [here](https://aristosmoe.blob.core.windows.net/artefacts/single_1x8_350M.nsys-rep)
- View in the Nsight GUI or 👇
- Run below to generate `single_1x8_350M_trace.txt` 
    ```shell
    nsys stats --filter-time="4s420ms/35s390ms" -r cuda_gpu_trace --timeunit usec --format column --output @"grep -E (Start*|ncclKernel_SendRecv_RING*)" single_1x8_350M.nsys-rep
    ```
- Run below to generate `single_1x8_filtered_sum.txt`
    ```shell
    nsys stats --filter-time="4s420ms/35s390ms" -r cuda_gpu_sum --timeunit usec --format column single_1x8_350M.nsys-rep
    ```
  
## Multi-Node 8x4
- Download 1.3B Trace from [here](https://aristosmoe.blob.core.windows.net/artefacts/multi_8x4_1.3B.nsys-rep)
- Download 350M Trace from [here](https://aristosmoe.blob.core.windows.net/artefacts/multi_8x4_350M.nsys-rep)
- View in the Nsight GUI or 👇
- Run below to generate `multi_8x4_1.3B_trace.txt` 
    ```shell
    nsys stats --filter-time="3s720ms/12s450ms" -r cuda_gpu_trace --timeunit usec --format column --output @"grep -E (Start*|ncclKernel_SendRecv_RING*)" multi_8x4_1.3B.nsys-rep
    ```
- Run below to generate `multi_8x4_1.3B_sum.txt`
    ```shell
    nsys stats --filter-time="3s720ms/12s450ms" -r cuda_gpu_sum --timeunit usec --format column multi_8x4_1.3B.nsys-rep
    ```
- Run below to generate `multi_8x4_350M_trace.txt` 
    ```shell
    nsys stats --filter-time="0s510ms/13s870ms" -r cuda_gpu_trace --timeunit usec --format column --output @"grep -E (Start*|ncclKernel_SendRecv_RING*)" multi_8x4_350M.nsys-rep
    ```
- Run below to generate `multi_8x4_350M_sum.txt`
    ```shell
    nsys stats --filter-time="0s510ms/13s870ms" -r cuda_gpu_sum --timeunit usec --format column multi_8x4_350M.nsys-rep
    ```
  
## Single Node Profiling
We profiled using the below command. You can change delay or duration.
```shell
nsys profile -s none --delay 200 --duration 40 --cpuctxsw none -t cuda,nvtx,cudnn,cublas,cusparse --cuda-graph-trace=node <training_script_name>
```

## Perlmutter Profiling
For multi-node training, we used the below script and executed as: `srun /bin/bash <the below script>`
```shell
#!/bin/bash
SCRIBE=1 # not node 0.
if [ "${SLURM_PROCID}" -eq "${SCRIBE}" ]; then
        echo "Node ${SLURM_PROCID} will profile!"
        nsys profile --kill none -s none --delay 120 --duration 15 --cpuctxsw none -t cuda,nvtx,cudnn,cublas,cusparse --cuda-graph-trace=node -o report_${SLURM_PROCID} <training_script_name>
else
        echo "Node ${SLURM_PROCID} will NOT profile!"
        <training_script_name>
fi
```