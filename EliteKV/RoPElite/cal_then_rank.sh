for (( fix=0; fix<=32; fix+=1 ))
do
if [ "$fix" -eq 0 ]; then
    python /data1/ssr/EliteKV/RoPElite/rank_0.py \
        --model_path /data1/ssr/model/Llama-2-7b-hf \
        --save_dir RoPElite/rank
fi
python RoPElite/cal_attn_distance.py \
    --model_path /data1/ssr/model/Llama-2-7b-hf \
    --task RoPElite \
    --fixed_dim_num $fix \
    --eval_iters 10 \
    --rank_file RoPElite/rank/Llama-2-7b-hf/RoPElite_$fix.pkl \
    --save_dir RoPElite/result
python RoPElite/post_process.py \
    --file_path  RoPElite/result/Llama-2-7b-hf/RoPElite_$fix.pkl \
    --save_dir RoPElite/rank
echo "Round$fix - Finished"
done
echo "All tasks has been finished."
