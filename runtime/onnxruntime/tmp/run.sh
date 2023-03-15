#下载模型
wget https://paddlespeech.bj.bcebos.com/speechx/wenet/wenetspeech/u2pp/20230315_u2%2B%2B_conformer_onnx.tar.gz
tar xzvf 20230315_u2++_conformer_onnx.tar.gz

#测试
export GLOG_logtostderr=1
export GLOG_v=2
wav_scp=wav.scp
onnx_dir=onnx_model_cpu
units=units.txt  # Change it to your model units path
../build/bin/decoder_main \
	--chunk_size 16 \
        --ctc_weight 0.5 \
        --reverse_weight 0.3 \
        --rescoring_weight 1.0 \
	--wav_scp $wav_scp \
	--onnx_dir $onnx_dir \
	--unit_path $units \
        --result log

onnx_dir=onnx_model_cpu_quant
units=units.txt  # Change it to your model units path
../build/bin/decoder_main \
        --chunk_size 16 \
        --ctc_weight 0.5 \
        --reverse_weight 0.3 \
        --rescoring_weight 1.0 \
        --wav_scp $wav_scp \
        --onnx_dir $onnx_dir \
        --unit_path $units \
        --result log.quant

#计算wer
python compute-wer.py --char=1 --v=1 text.truth log > wer
python compute-wer.py --char=1 --v=1 text.truth log.quant > wer.quant
