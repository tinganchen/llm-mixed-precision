# mpl

for SIZE in llama-7b #llama-30b llama-7b
    do
    for BIT in 16 8 6 4 
        do

python main.py --job_dir ../../../llm_experiment/${SIZE}/mpl/t_${BIT}_1/ \
    --search_dir ../../../serach_experiment/${SIZE}/mpl/t_${BIT}_0/ \
    --bitW $BIT --abitW $BIT \
    --qmethod mpl \
    --model $SIZE --target_model LlamaForCausalLM \
    --max_seq_len 2048 \
    --eval_batch_size 4 \
    --finetuned False \
    --source_file ../../../llm_experiment/gpt2/all/t_0/ \
    --tasks all \
    --test_only True \
    --port 29519 \
    --gpus 3,2,1
    
        done
    done
    
# mpl - fix_ab    
for SIZE in llama-13b
    do
        for WBIT in 8 6 4
        do
        for ABIT in 16 8 6 4 
            do
        python main.py \
            --job_dir ../../../llm_experiment/${SIZE}/mpl/t_m${WBIT}_f${ABIT}_1/ \
            --search_dir ../../../serach_experiment/${SIZE}/mpl/t_${WBIT}_0/ \
            --model $SIZE --target_model LlamaForCausalLM \
            --max_seq_len 2048 \
            --eval_batch_size 4 \
            --bitW $WBIT --abitW $ABIT \
            --fix_bitW False --fix_abitW True \
            --qmethod mpl \
            --source_file ../../../llm_experiment/gpt2/all/t_0/ \
            --finetuned False \
            --test_only True \
            --tasks all \
            --port 29529 \
            --gpus 3,2,1,0,4,5
            done
        done
    done
