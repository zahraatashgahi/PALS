#!/bin/bash
#SBATCH -J ILI

train_dense="yes" ; 
train_granet="no";
train_PALS="yes";


for seed in  2020 #2021 2022
do
	for model in "ns_Transformer" #"Autoformer"  "Transformer" "Informer" "FEDformer" 
	do
		for pred_len in  24 36 48 60 
		do 
			if [ "$pred_len" -eq 24 ]; then 
				p_hidden_dims=32
			elif [ "$pred_len" -eq 36 ]
			then 
				p_hidden_dims=32
			elif [ "$pred_len" -eq 48 ]
			then 
				p_hidden_dims=16
			elif [ "$pred_len" -eq 60 ]
			then 
				p_hidden_dims=8
			fi
			
			
			
			#################################  Adapt-Tune  #################################
			if [ "$train_PALS" = "yes" ]; then 
				for method in  "PALS" 
				do
					python -u run.py \
					  --is_training 1 \
					  --root_path ../dataset/illness/ \
					  --data_path national_illness.csv \
					  --model_id "ili_36_"$pred_len \
					  --model $model \
					  --data custom \
					  --features M \
					  --seq_len 36 \
					  --label_len 18 \
					  --pred_len $pred_len \
					  --e_layers 2 \
					  --d_layers 1 \
					  --factor 3 \
					  --enc_in 7 \
					  --dec_in 7 \
					  --c_out 7 \
					  --des 'Exp_h'$p_hidden_dims'_l2' \
					  --p_hidden_dims $p_hidden_dims $p_hidden_dims \
					  --p_hidden_layers 2 \
					  --seed $seed\
					  --sparse --method $method --update-frequency 5\
					  --init-density 1 --sparse-init 'ERK' 
				done
			fi
	
		


			
			#################################  GraNet #################################
			if [ "$train_granet" = "yes" ]; then 
				for final_density in 0.75 0.5 0.35 0.2 0.1 0.05 
				do
					python -u run.py \
					  --is_training 1 \
					  --root_path ../dataset/illness/ \
					  --data_path national_illness.csv \
					  --model_id "ili_36_"$pred_len \
					  --model $model \
					  --data custom \
					  --features M \
					  --seq_len 36 \
					  --label_len 18 \
					  --pred_len $pred_len \
					  --e_layers 2 \
					  --d_layers 1 \
					  --factor 3 \
					  --enc_in 7 \
					  --dec_in 7 \
					  --c_out 7 \
					  --des 'Exp_h'$p_hidden_dims'_l2' \
					  --p_hidden_dims $p_hidden_dims $p_hidden_dims \
					  --p_hidden_layers 2 \
					  --seed $seed\
					  --sparse --method 'GraNet' --update-frequency 5\
					  --init-density 1 --sparse-init 'ERK' --final-density $final_density 
				done
			fi			
			

			#################################   dense #################################
			if [ "$train_dense" = "yes" ]; then 
				python -u run.py \
				  --is_training 1 \
				  --root_path ../dataset/illness/ \
				  --data_path national_illness.csv \
				  --model_id "ili_36_"$pred_len \
				  --model $model \
				  --data custom \
				  --features M \
				  --seq_len 36 \
				  --label_len 18 \
				  --pred_len $pred_len \
				  --e_layers 2 \
				  --d_layers 1 \
				  --factor 3 \
				  --enc_in 7 \
				  --dec_in 7 \
				  --c_out 7 \
				  --des 'Exp_h'$p_hidden_dims'_l2' \
				  --p_hidden_dims $p_hidden_dims $p_hidden_dims \
				  --p_hidden_layers 2 \
				  --seed $seed 
			  
			fi  
						
		done
	done

done			  
