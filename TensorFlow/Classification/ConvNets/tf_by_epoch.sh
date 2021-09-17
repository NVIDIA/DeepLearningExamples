for (( i = 0; i < 91; i++ )); do
    mkdir /data/tfrecords/epoch_"${i}"/
    echo "Created Folder epoch_${i}"
    mpiexec --allow-run-as-root --bind-to socket -np 2 python3 main.py --arch=resnet50 --mode=train_and_evaluate --num_iter=i --batch_size=192 --warmup_steps=0 --lr_warmup_epochs=0 --model_dir=/data/tfrecords/amp_model     --data_dir=/data/tfrecords/tfrecords --data_idx_dir=/data/tfrecords/dali_idx     --results_dir=/data/tfrecords/epoch_"${i}"/results --export_dir=/data/tfrecords/best_model --weight_init=fan_in --amp --log_filename epoch_"${i}".json
    echo "Epoch ${i} done"
done
