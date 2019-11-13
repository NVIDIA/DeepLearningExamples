bash test_infer.sh -bs 1 -il 128 -p amp --num-iters 1003 --tacotron2 checkpoint_Tacotron2_amp --waveglow checkpoint_WaveGlow_amp
bash test_infer.sh -bs 4 -il 128 -p amp --num-iters 1003 --tacotron2 checkpoint_Tacotron2_amp --waveglow checkpoint_WaveGlow_amp
bash test_infer.sh -bs 1 -il 128 -p fp32 --num-iters 1003 --tacotron2 checkpoint_Tacotron2_fp32 --waveglow checkpoint_WaveGlow_fp32
bash test_infer.sh -bs 4 -il 128 -p fp32 --num-iters 1003 --tacotron2 checkpoint_Tacotron2_fp32 --waveglow checkpoint_WaveGlow_fp32

