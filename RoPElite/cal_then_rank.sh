for (( fix=0; fix<=32; fix+=1 ))
do
if [ "$fix" -eq 0 ]; then
    python RoPElite/rank_0.py \
        --model_path path/to/your/model \
        --save_dir RoPElite/rank
fi
python RoPElite/cal_attn_distance.py \
    --model_path path/to/your/model \
    --data_path path/to/your/data \
    --task RoPElite \
    --fixed_dim_num $fix \
    --eval_iters 2000 \
    --rank_file RoPElite/rank/model_name/RoPElite_$fix.pkl \
    --save_dir RoPElite/result
python RoPElite/post_process.py \
    --file_path  RoPElite/result/model_name/RoPElite_$fix.pkl \
    --save_dir RoPElite/rank
echo "Round$fix - Finished"
done
echo "All tasks has been finished."
