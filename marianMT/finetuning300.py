import torch
import pandas as pd
import time
from datasets import Dataset
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

# Marcar o início do tempo
start_time = time.time()

# Verificar se há GPU disponível
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

if device == "cuda":
    torch.cuda.empty_cache()
    print("Memória da GPU liberada.")

# Caminhos dos arquivos CSV
train_csv_path = "../poemas/poemas300/train/frances_ingles_train.csv"
val_csv_path = "../poemas/poemas300/validation/frances_ingles_validation.csv"

# Carregar os dados dos CSVs como Dataset Hugging Face
def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return Dataset.from_pandas(df)
    except FileNotFoundError:
        print(f"Erro: O arquivo {csv_path} não foi encontrado.")
        raise
    except Exception as e:
        print(f"Erro ao carregar o arquivo {csv_path}: {e}")
        raise

try:
    train_dataset = load_data(train_csv_path)
    val_dataset = load_data(val_csv_path)
except Exception as e:
    print(f"Falha ao carregar os dados. Erro: {e}")
    exit(1)

# Escolher o modelo base do MarianMT
try:
    model_name = "Helsinki-NLP/opus-mt-fr-en"  # Troque pelo idioma correto
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
except Exception as e:
    print(f"Erro ao carregar o modelo ou tokenizer: {e}")
    exit(1)

# Função de pré-processamento dos textos
def preprocess_function(examples):
    try:
        inputs = tokenizer(examples["original_poem"], padding="max_length", truncation=True, max_length=512)  # Ajuste o max_length
        targets = tokenizer(examples["translated_poem"], padding="max_length", truncation=True, max_length=512)
        inputs["labels"] = targets["input_ids"]
        return inputs
    except Exception as e:
        print(f"Erro durante o pré-processamento: {e}")
        raise

# Aplicar o pré-processamento
try:
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)
except Exception as e:
    print(f"Erro ao aplicar o pré-processamento: {e}")
    exit(1)

# Configurar os parâmetros do treinamento
try:
    training_args = Seq2SeqTrainingArguments(
        output_dir="/home/ubuntu/finetuning/marianMT/marianMT_frances_ingles",
        eval_strategy="epoch",
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
except Exception as e:
    print(f"Erro ao configurar os parâmetros de treinamento: {e}")
    exit(1)

# Criar o trainer
try:
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
except Exception as e:
    print(f"Erro ao criar o trainer: {e}")
    exit(1)


class LossCallback:
    def __init__(self):
        self.last_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.last_loss = logs["loss"]

loss_callback = LossCallback()
trainer.add_callback(loss_callback)

# Iniciar o treinamento
try:
    trainer.train()
except Exception as e:
    print(f"Erro durante o treinamento: {e}")
    exit(1)

last_loss = None
for log in trainer.state.log_history:
    if "loss" in log:
        last_loss = log["loss"]

if last_loss is not None:
    print(f"Última loss registrada: {last_loss:.4f}")
else:
    print("Nenhuma loss registrada.")

# Salvar o modelo treinado
try:
    model.save_pretrained("/home/ubuntu/finetuning/marianMT/marianMT_frances_ingles")
    tokenizer.save_pretrained("/home/ubuntu/finetuning/marianMT/marianMT_frances_ingles")
    print("Fine-tuning finalizado e modelo salvo.")
except Exception as e:
    print(f"Erro ao salvar o modelo: {e}")
    exit(1)

# Calcular o tempo de execução
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tempo total de execução: {elapsed_time:.2f} segundos")

print(f"Tamanho do dataset de treino: {len(train_dataset)}")
print(f"Tamanho do dataset de validação: {len(val_dataset)}")
print(f"Última loss registrada: {loss_callback.last_loss:.4f}" if loss_callback.last_loss else "Nenhuma loss registrada.")
