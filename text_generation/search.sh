# llama
    
## mpl    
    
for SIZE in  llama-13b #llama-7b llama-30b
    do
    for BIT in 8 6 
        do
python3 search.py \
    --model $SIZE --target_model LlamaForCausalLM \
    --job_dir ../../../serach_experiment/${SIZE}/mpl_m/t_${BIT}_0/ \
    --method all --max_seq_len 2048 \
    --qmethod mpl \
    --bitW $BIT --abitW $BIT \
    --source_file ../../../llm_experiment/gpt2/all/t_0/ \
    --finetuned False \
    --bit_search_only True \
    --port 29518 \
    --gpus -1 \
    --eval_batch_size 1
    
        done
    done

