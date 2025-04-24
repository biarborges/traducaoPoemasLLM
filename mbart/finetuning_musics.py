import torch
import pandas as pd
import time
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from datasets import Dataset, concatenate_datasets
from transformers import TrainingArguments, Trainer
from tqdm.auto import tqdm
import os

# Configurar ambiente para liberar memória GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

start_time = time.time()

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Caminhos dos arquivos CSV
poemas_train = "../poemas/train/frances_ingles_train.csv"
poemas_val = "../poemas/validation/frances_ingles_validation.csv"
musicas_train = "../musicas/train/frances_ingles_train.csv"
musicas_val = "../musicas/validation/frances_ingles_validation.csv"

# Carregar os dados
df_poemas_train = pd.read_csv(poemas_train).dropna()
df_poemas_val = pd.read_csv(poemas_val).dropna()
df_musicas_train = pd.read_csv(musicas_train).dropna()
df_musicas_val = pd.read_csv(musicas_val).dropna()

# Converter para datasets Hugging Face
poemas_train_dataset = Dataset.from_pandas(df_poemas_train)
poemas_val_dataset = Dataset.from_pandas(df_poemas_val)
musicas_train_dataset = Dataset.from_pandas(df_musicas_train)
musicas_val_dataset = Dataset.from_pandas(df_musicas_val)

# Concatenar datasets de treino e validação
train_dataset = concatenate_datasets([poemas_train_dataset, musicas_train_dataset])
val_dataset = concatenate_datasets([poemas_val_dataset, musicas_val_dataset])

# Carregar modelo e tokenizer do mBART
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)

# Função para tokenizar os dados
def preprocess_function(examples):
    inputs = tokenizer(examples["original_poem"], max_length=64, truncation=True, padding="max_length")
    targets = tokenizer(examples["translated_poem"], max_length=64, truncation=True, padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs

# Tokenizar datasets com tqdm
print("Tokenizando dataset de treino...")
train_dataset = train_dataset.map(preprocess_function, batched=True, desc="Tokenizando treino", batch_size=32)

print("Tokenizando dataset de validação...")
val_dataset = val_dataset.map(preprocess_function, batched=True, desc="Tokenizando validação", batch_size=32)

# Definir parâmetros de treinamento
training_args = TrainingArguments(
    output_dir="/home/ubuntu/finetuning_fr_en",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4, #8 no artigo
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=20, #10 no artigo
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    save_total_limit=1,
    fp16=True,
    logging_dir="./logs",
    logging_steps=50,
    report_to="none"
)

# Criar Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=None,
)

# Iniciar o treinamento
trainer.train()

# Salvar modelo treinado
model.save_pretrained("/home/ubuntu/finetuning_fr_en")
tokenizer.save_pretrained("/home/ubuntu/finetuning_fr_en")

print("Fine-tuning concluído e modelo salvo.")

# Tempo total de execução
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tempo total de execução: {elapsed_time:.2f} segundos")
