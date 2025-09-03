import torch
import pandas as pd
import time
from datasets import Dataset, concatenate_datasets
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

start_time = time.time()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

if device == "cuda":
    torch.cuda.empty_cache()
    print("Memória da GPU liberada.")

# Caminhos dos arquivos
#poem_train_csv = "../poemas/train/frances_ingles_train.csv"
#poem_val_csv = "../poemas/validation/frances_ingles_validation.csv"
music_train_csv = "../musicas/train/frances_ingles_musics_train.csv"
music_val_csv = "../musicas/validation/frances_ingles_musics_validation.csv"

model = "/home/ubuntu/finetuning_fr_en"
tokenizer = "/home/ubuntu/finetuning_fr_en"

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    return Dataset.from_pandas(df)

# Carregar os datasets individualmente
try:
    #poem_train = load_dataset(poem_train_csv)
    music_train = load_dataset(music_train_csv)
    #poem_val = load_dataset(poem_val_csv)
    music_val = load_dataset(music_val_csv)
except Exception as e:
    print(f"Erro ao carregar datasets: {e}")
    exit(1)

# Concatenar os conjuntos de poemas + músicas
train_dataset = concatenate_datasets([music_train])
val_dataset = concatenate_datasets([music_val])

#train_dataset = concatenate_datasets([poem_train, music_train])
#val_dataset = concatenate_datasets([poem_val, music_val])

# Carregar modelo e tokenizer
model_name = "Helsinki-NLP/opus-mt-ROMANCE-en"  
try:
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
except Exception as e:
    print(f"Erro ao carregar modelo/tokenizer: {e}")
    exit(1)

# Pré-processamento
def preprocess_function(examples):
    try:
        inputs = tokenizer(examples["original_poem"], padding="max_length", truncation=True, max_length=512)
        targets = tokenizer(examples["translated_poem"], padding="max_length", truncation=True, max_length=512)
        inputs["labels"] = targets["input_ids"]
        return inputs
    except Exception as e:
        print(f"Erro no pré-processamento: {e}")
        raise

try:
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)
except Exception as e:
    print(f"Erro ao aplicar tokenizer: {e}")
    exit(1)

# Argumentos de treinamento
training_args = Seq2SeqTrainingArguments(
    output_dir="/home/ubuntu/finetuning_fr_en",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    save_strategy="epoch",
    report_to="none"
)

# Treinador
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

# Treinar
try:
    trainer.train()
except Exception as e:
    print(f"Erro no treinamento: {e}")
    exit(1)

# Salvar modelo
try:
    model.save_pretrained(model)
    tokenizer.save_pretrained(tokenizer)
    print("Modelo salvo com sucesso.")
except Exception as e:
    print(f"Erro ao salvar: {e}")
    exit(1)

elapsed_time = time.time() - start_time
print(f"Tempo total de execução: {elapsed_time:.2f} segundos")
print(f"Tamanho do dataset de treino: {len(train_dataset)}")
print(f"Tamanho do dataset de validação: {len(val_dataset)}")
