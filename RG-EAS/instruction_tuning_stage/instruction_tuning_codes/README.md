For the instruction tuning stage, we reference the code from the open-source GitHub project [MedicalGPT](https://github.com/shibing624/MedicalGPT). 

To run it, you first should install the required environment:
```
pip install -r requirements.txt --upgrade
```

Next, please run the script of fine-tuning:
```
bash run_sft.sh
```

Note that if you are using LoRA fine-tuning (as in this paper), 
you should merge the LoRA weights (i.e., the output from the previous step) into your base model:
```
bash run_merge_peft_adapter.sh
```

The merged model is saved in the `--output` directory.
