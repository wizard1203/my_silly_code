rem python get_caffe_performance.py -d p100_ipc_gemm
rem python get_power.py -d p100_ipc_gemm
rem python get_data.py -d p100_ipc_gemm

rem python get_caffe_performance.py -d p100_fft_tile
rem python get_power.py -d p100_fft_tile
rem python get_data.py -d p100_fft_tile

rem python get_caffe_performance.py -d p100_winograd_nonfused
rem python get_power.py -d p100_winograd_nonfused
rem python get_data.py -d p100_winograd_nonfused

rem python get_caffe_performance.py -d v100_ipc_gemm
rem python get_power.py -d v100_ipc_gemm
rem python get_data.py -d v100_ipc_gemm

rem python get_caffe_performance.py -d v100_fft_tile
rem python get_power.py -d v100_fft_tile
rem python get_data.py -d v100_fft_tile

rem python get_caffe_performance.py -d v100_winograd
rem python get_power.py -d v100_winograd
rem python get_data.py -d v100_winograd

rem python get_caffe_performance.py -d gtx2080ti_ipc_gemm
rem python get_power.py -d gtx2080ti_ipc_gemm
rem python get_gtx2080ti_data.py -d gtx2080ti_ipc_gemm

rem python get_caffe_performance.py -d gtx2080ti_fft_tile
rem python get_power.py -d gtx2080ti_fft_tile
rem python get_gtx2080ti_data.py -d gtx2080ti_fft_tile

rem python get_caffe_performance.py -d gtx2080ti_winograd_nonfused
rem python get_power.py -d gtx2080ti_winograd_nonfused
rem python get_gtx2080ti_data.py -d gtx2080ti_winograd_nonfused


python get_inference_performance.py -d gtx2080ti_auto
python get_inference_power.py -d gtx2080ti_auto -c 100000
python get_gtx2080ti_inference_data.py -d gtx2080ti_auto


python get_inference_performance.py -d v100_auto
python get_inference_power.py -d v100_auto -c 45000
python get_inference_data.py -d v100_auto

python get_inference_performance.py -d p100_auto
python get_inference_power.py -d p100_auto -c 45000
python get_inference_data.py -d p100_auto


