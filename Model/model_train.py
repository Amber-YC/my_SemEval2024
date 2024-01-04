from sklearn.model_selection import train_test_split
from preprocessing import load_data, get_batches
from adapters_model import get_word_embeddings, encode_batch, set_lang_adapter
import numpy as np
import torch.nn as nn
import torch
import adapters.composition as ac
from transformers import TrainingArguments, EvalPrediction


"""load english training and eval data"""
tracka_eng = '../Semantic_Relatedness_SemEval2024-main/Track A/eng/eng_train.csv'
eng_data = load_data(tracka_eng)
eng_batches = list(get_batches(10, eng_data))
print(eng_batches)
eng_adapter = set_lang_adapter("en/wiki@ukp")
for batch in eng_batches[:5]:
    batch_tokens = encode_batch(batch)
    batch_output = get_word_embeddings(batch_tokens, eng_adapter)
    print(batch_output)

# texts_train, texts_val, labels_train, labels_val = train_test_split(train_pairs_eng, train_scores_eng, test_size=0.2, random_state=42)

"""load arb language data"""
trackc_arb_dev = '../Semantic_Relatedness_SemEval2024-main/Track C/arb/arb_dev.csv'
arb_data = load_data(trackc_arb_dev)

"""load amh language data"""
trackc_amh_dev = '../Semantic_Relatedness_SemEval2024-main/Track C/amh/amh_dev.csv'
amh_data = load_data(trackc_arb_dev)

"""load ind language data"""
trackc_ind_dev = '../Semantic_Relatedness_SemEval2024-main/Track C/ind/ind_dev.csv'
ind_data = load_data(trackc_ind_dev)



# training_args = TrainingArguments(
#     learning_rate=1e-4,
#     num_train_epochs=6,
#     per_device_train_batch_size=32,
#     per_device_eval_batch_size=32,
#     logging_steps=200,
#     output_dir="./training_output",
#     overwrite_output_dir=True,
#     # The next line is important to ensure the dataset labels are properly passed to the model
#     remove_unused_columns=False,
# )
#
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# loss_fn = torch.nn.MSELoss()
#
# def compute_metrics(p: EvalPrediction):
#     return {"mse_loss": loss_fn(torch.tensor(p.predictions), torch.tensor(p.label_ids)).item()}
#
# trainer = AdapterTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     compute_metrics=compute_metrics,
# )
#
# trainer.train()
#
# # 评估模型
# trainer.evaluate()

