import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

# Verificar se há GPU disponível
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Caminhos dos arquivos CSV
train_csv_path = "../traducaoPoemasLLM/poemas/train/frances_ingles_train.csv"
val_csv_path = "../traducaoPoemasLLM/poemas/validation/frances_ingles_validation.csv"

# Carregar os dados dos CSVs como Dataset Hugging Face
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return Dataset.from_pandas(df)

train_dataset = load_data(train_csv_path)
val_dataset = load_data(val_csv_path)

# Escolher o modelo base do MarianMT
model_name = "Helsinki-NLP/opus-mt-ROMANCE-EN"  # Troque pelo idioma correto
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(device)

# Função de pré-processamento dos textos
def preprocess_function(examples):
    inputs = tokenizer(examples["original_poem"], padding="max_length", truncation=True, max_length=512)  # Ajuste o max_length
    targets = tokenizer(examples["translated_poem"], padding="max_length", truncation=True, max_length=512)
    inputs["labels"] = targets["input_ids"]
    return inputs


# Aplicar o pré-processamento
train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# Configurar os parâmetros do treinamento
training_args = Seq2SeqTrainingArguments(
    output_dir="../fineTuning/marianMT",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),  # Usa FP16 se GPU suportar
    save_strategy="epoch",
    report_to="none"  # Evita logs desnecessários
)

# Criar o trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

# Iniciar o treinamento
trainer.train()

# Salvar o modelo treinado
model.save_pretrained("../fineTuning/marianMT_frances_ingles")
tokenizer.save_pretrained("../fineTuning/marianMT_frances_ingles")

print("Fine-tuning finalizado e modelo salvo.")
