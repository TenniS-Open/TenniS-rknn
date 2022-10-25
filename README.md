RKNN

Support single input and multi outputs.

The default config channel_mean_value='0 0 0 1', reorder_channel='0 1 2'.
If not set, 3 channels image will apply channel_mean_value = '0 0 0 255' on input.

Keng:
1. Gemm only support TransA=0, TransB=1
2. Gather not supported. also DepthToSpace and SpaceToDepth.
3. broadcast sub not supported in 1.3.1 now. what?

2020-5-21
