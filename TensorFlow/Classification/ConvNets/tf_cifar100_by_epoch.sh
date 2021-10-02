# nvidia-docker run --rm -it --ipc=host -v /work/chauhans/cifar100:/cifar100 rn50_tf1

# set classes to 100 in main.py before running

for (( n = 0; i < 6; n++ )); do
  for (( i = 0; i < 91; i++ )); do
      mkdir /cifar100/TensorFlow/run_"${n}"/epoch_"${i}"/
      echo "Created Folder epoch_${i}"
      mpiexec --allow-run-as-root --bind-to socket -np 2 python3 main.py --arch=resnet50 --mode=train_and_evaluate \
      --num_iter="${i}" --batch_size=192 --warmup_steps=0 --lr_warmup_epochs=0 --model_dir=/cifar100/TensorFlow/run_"${n}"/model     \
      --data_dir=/cifar100/tfrecords  --results_dir=/cifar100/TensorFlow/run_"${n}"/epoch_"${i}"/          \
      --export_dir=/cifar100/TensorFlow/run_"${n}"/model --weight_init=fan_in --amp --log_filename run_"${n}"epoch_"${i}".json
      echo "Epoch ${i} done"
  done
  echo "Run ${n} done"
done


#      mpiexec --allow-run-as-root --bind-to socket -np 2 python3 main.py --arch=resnet50 --mode=train_and_evaluate \
#      --num_iter=90 --batch_size=192 --warmup_steps=0 --lr_warmup_epochs=0 --model_dir=/cifar100/TensorFlow/model     \
#      --data_dir=/cifar100/tfrecords  --results_dir=/cifar100/TensorFlow/test          \
#      --export_dir=/cifar100/TensorFlow/model --weight_init=fan_in --amp --log_filename test.json