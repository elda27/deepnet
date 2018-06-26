
set TRUTH_DIR=F:\dataset\2018-05v2\DRR
set UNET_DIR=X:\kabashima\RibSegmentation\logs\logs\Ribcage\use_id-list_trial-0_test-0\unet_2d_VCS-42e08b_TIME-20180604-182732\tests_merged
set TL_NET_ROOT=X:\kabashima\tl-net\logs\100-TIME-20180624-045240

python tools\compute_metric.py --gpu 0 --truth-dir %TRUTH_DIR% --test-dir %UNET_DIR% %TL_NET_ROOT%\test_stage1 %TL_NET_ROOT%\test_stage2 %TL_NET_ROOT%\test_stage3 --test-index .\ini\id-list_trial-0_test-0.txt --test-names U_net ConvAE Segnet TL_net --output metric.nc --num-parallel 5 
