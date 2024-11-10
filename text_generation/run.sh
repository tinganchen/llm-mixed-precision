# mpl - fix_ab    
for SIZE in llama-13b # llama-7b
    do
        for WBIT in 8 6 4
        do
        for ABIT in 16 8 6 4
            do
        python main.py \
            --job_dir ../../../llm_experiment/${SIZE}/mpl/tg3_m${WBIT}_f${ABIT}_1/ \
            --search_dir ../../../serach_experiment/${SIZE}/mpl/t_${WBIT}_0/ \
            --model $SIZE --target_model LlamaForCausalLM \
            --max_seq_len 2048 \
            --eval_batch_size 2 \
            --bitW $WBIT --abitW $ABIT \
            --fix_bitW False --fix_abitW True \
            --qmethod mpl \
            --source_file ../../../llm_experiment/gpt2/all/t_0/ \
            --finetuned False \
            --test_only True \
            --tasks all \
            --port 29537 \
            --gpus -1
            done
        done
    done

