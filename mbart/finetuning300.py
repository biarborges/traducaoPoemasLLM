import torch
import pandas as pd
import time
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from datasets import Dataset
from transformers import TrainingArguments, Trainer
import os
from torch.utils.data import DataLoader

# Configurar ambiente para liberar memória GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

start_time = time.time()

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Caminhos dos arquivos CSV
train_file = "../poemas/poemas300/train/frances_ingles_train.csv"
val_file = "../poemas/poemas300/validation/frances_ingles_validation.csv"

# Carregar os dados
df_train = pd.read_csv(train_file).dropna()
df_val = pd.read_csv(val_file).dropna()

# Converter para dataset Hugging Face
train_dataset = Dataset.from_pandas(df_train)
val_dataset = Dataset.from_pandas(df_val)

# Carregar modelo e tokenizer do mBART
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)

# Função para tokenizar os dados
def preprocess_function(examples):
    inputs = tokenizer(examples["original_poem"], max_length=256, truncation=True, padding="max_length")
    targets = tokenizer(examples["translated_poem"], max_length=256, truncation=True, padding="max_length")

    inputs["labels"] = targets["input_ids"]  # Definir os labels para o modelo aprender
    return inputs

# Tokenizar datasets de treino e validação
train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# Criar DataLoader personalizados para o treinamento e validação
train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=2, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, num_workers=2, pin_memory=True)

# Definir parâmetros de treinamento
training_args = TrainingArguments(
    output_dir="/home/ubuntu/finetuning_fr_ing",
    evaluation_strategy="epoch",  # Avaliar ao final de cada época
    save_strategy="epoch",  # Salvar modelo ao final de cada época
    per_device_train_batch_size=4,  # Ajuste conforme memória disponível
    per_device_eval_batch_size=8,  # Ajuste conforme necessário
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    save_total_limit=1,
    fp16=True,
    logging_dir="./logs",
    logging_steps=50,
    report_to="none"
)

# Criar Trainer com DataLoader personalizado
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=None,  # Colocar seu próprio DataLoader aqui
)

# Iniciar o treinamento
trainer.train()

# Salvar modelo treinado
model.save_pretrained("/home/ubuntu/finetuning_fr_ing")
tokenizer.save_pretrained("/home/ubuntu/finetuning_fr_ing")

print("Fine-tuning concluído e modelo salvo.")

# Tempo total de execução
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tempo total de execução: {elapsed_time:.2f} segundos")
