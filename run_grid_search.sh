#/usr/bin/bash
declare -a hidden=(128 512 1024)
declare -a batch=(32 128)
declare -a depth=(1 3)
declare -a sequence=(30 100 200)
for sequence in ${sequence[@]}
do
    for depth in ${depth[@]}
    do
	for batch in ${batch[@]}
	do
	    for hidden in ${hidden[@]}
	    do
		echo "-----------------------"
		echo $sequence
		echo $depth
		echo $batch
		echo $hidden
		echo "-----------------------"
		THEANO_FLAGS="mode=FAST_RUN,device=cuda,floatX=float32" python cudnn_rnn.py -n lstm -d $depth -b $batch -o $hidden -t $sequence
	    done
	done
    done
done

echo '=========================================='
echo "SWITCHING TO DEFAULT RNN"
echo '=========================================='

for sequence in ${sequence[@]}
do
    for depth in ${depth[@]}
    do
	for batch in ${batch[@]}
	do
	    for hidden in ${hidden[@]}
	    do
		echo "-----------------------"
		echo $sequence
		echo $depth
		echo $batch
		echo $hidden
		echo "-----------------------"
		THEANO_FLAGS="mode=FAST_RUN,device=cuda,floatX=float32" python rnn.py -n lstm -d $depth -b $batch -o $hidden -t $sequence
	    done
	done
    done
done

