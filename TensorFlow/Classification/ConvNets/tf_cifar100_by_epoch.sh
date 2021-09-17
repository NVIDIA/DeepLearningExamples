# nvidia-docker run --rm -it --ipc=host -v /work/chauhans/cifar100:/cifar100 rn50_tf1

for (( n = 0; i < 1; n++ )); do
  for (( i = 0; i < 91; i++ )); do
      mkdir /imagenet/TensorFlow/run_"${n}"/epoch_"${i}"/
      echo "Created Folder epoch_${i}"
      mpiexec --allow-run-as-root --bind-to socket -np 2 python3 main.py --arch=resnet50 --mode=train_and_evaluate \
      --num_iter=i --batch_size=192 --warmup_steps=0 --lr_warmup_epochs=0 --model_dir=/imagenet/TensorFlow/model     \
      --data_dir=/imagenet/tfrecords  --results_dir=/imagenet/TensorFlow/run_"${n}"/epoch_"${i}"/          \
      --export_dir=/imagenet/TensorFlow/model --weight_init=fan_in --amp --log_filename run_"${n}"epoch_"${i}".json
      echo "Epoch ${i} done"
  done
  echo "Run ${n} done"
done
