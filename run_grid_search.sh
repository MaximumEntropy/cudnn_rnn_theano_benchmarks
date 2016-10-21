#/usr/bin/bash
for sequence in 30 100 200
do
    for depth in 1 3
    do
		for batch in 32 128
		do
		    for hidden in 128 512 1024
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
echo "SWITCHING TO DEFAULT FastLSTM"
echo '=========================================='

for sequence in 30 100 200
do
    for depth in 1 3
    do
		for batch in 32 128
		do
		    for hidden in 128 512 1024
		    do
				echo "-----------------------"
				echo $sequence
				echo $depth
				echo $batch
				echo $hidden
				echo "-----------------------"
				THEANO_FLAGS="mode=FAST_RUN,device=cuda,floatX=float32" python rnn.py -n fastlstm -d $depth -b $batch -o $hidden -t $sequence
		    done
		done
    done
done
