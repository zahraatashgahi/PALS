#!/bin/bash


train_dense="yes"; 
train_granet="no";
train_PALS="yes"; 


for seed in  2020 
do
	for model in "ns_Transformer" # "Autoformer" "Transformer" "Informer"  "FEDformer" 
	do
	
		for pred_len in 96 192 336 720
		do 
			if [ "$pred_len" -eq 96 ]; then 
				p_hidden_dims=256
			elif [ "$pred_len" -eq 192 ]
			then 
				p_hidden_dims=256
			elif [ "$pred_len" -eq 336 ]
			then 
				p_hidden_dims=128
			elif [ "$pred_len" -eq 720 ]
			then 
				p_hidden_dims=128
			fi
			
	
			
			#################################  PALS  #################################
			if [ "$train_PALS" = "yes" ]; then 
				for method in "PALS" 
				do
					python -u run.py \
					  --is_training 1 \
					  --root_path ../dataset/electricity/ \
					  --data_path electricity.csv \
					  --model_id "ECL_96_"$pred_len \
					  --model $model \
					  --data custom \
					  --features M \
					  --seq_len 96 \
					  --label_len 48 \
					  --pred_len $pred_len \
					  --e_layers 2 \
					  --d_layers 1 \
					  --factor 3 \
					  --enc_in 321 \
					  --dec_in 321 \
					  --c_out 321 \
					  --des 'Exp_h'$p_hidden_dims'_l2' \
					  --p_hidden_dims $p_hidden_dims $p_hidden_dims \
					  --p_hidden_layers 2\
					  --seed $seed \
					  --sparse --method $method  --update-frequency 20\
					  --init-density 1 --sparse-init 'ERK' 
				done
			fi
			

			
			#################################   dense #################################
			if [ "$train_dense" = "yes" ]; then 

				python -u run.py \
				  --is_training 1 \
				  --root_path ../dataset/electricity/ \
				  --data_path electricity.csv \
				  --model_id "ECL_96_"$pred_len \
				  --model $model \
				  --data custom \
				  --features M \
				  --seq_len 96 \
				  --label_len 48 \
				  --pred_len $pred_len \
				  --e_layers 2 \
				  --d_layers 1 \
				  --factor 3 \
				  --enc_in 321 \
				  --dec_in 321 \
				  --c_out 321 \
				  --des 'Exp_h'$p_hidden_dims'_l2' \
				  --p_hidden_dims $p_hidden_dims $p_hidden_dims \
				  --p_hidden_layers 2\
				  --seed $seed  
			fi
			#################################  GraNet #################################
			if [ "$train_granet" = "yes" ]; then 
				for final_density in 0.75 0.5 0.35 0.2 0.1 0.05
				do
					python -u run.py \
					  --is_training 1 \
					  --root_path ../dataset/electricity/ \
					  --data_path electricity.csv \
					  --model_id "ECL_96_"$pred_len \
					  --model $model \
					  --data custom \
					  --features M \
					  --seq_len 96 \
					  --label_len 48 \
					  --pred_len $pred_len \
					  --e_layers 2 \
					  --d_layers 1 \
					  --factor 3 \
					  --enc_in 321 \
					  --dec_in 321 \
					  --c_out 321 \
					  --des 'Exp_h'$p_hidden_dims'_l2' \
					  --p_hidden_dims $p_hidden_dims $p_hidden_dims \
					  --p_hidden_layers 2 \
					  --seed $seed \
					  --sparse --method 'GraNet' --update-frequency 20\
					  --init-density 1 --sparse-init 'ERK' --final-density $final_density
				done
			fi

		done
	done
done
