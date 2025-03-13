import torch
import pandas as pd
from datasets import Dataset
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

# Verificar se há GPU disponível
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Caminhos dos arquivos CSV
train_csv_path = "../poemas/train/frances_ingles_train.csv"
val_csv_path = "../poemas/validation/frances_ingles_validation.csv"

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
    model_name = "Helsinki-NLP/opus-mt-fr-en"  # Modelo para tradução de francês para inglês
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
except Exception as e:
    print(f"Erro ao carregar o modelo ou tokenizer: {e}")
    exit(1)

# Função para dividir poemas em partes menores
def split_poem(poem, max_tokens=512):
    # Tokenizar o poema sem adicionar tokens especiais (como [CLS] ou [SEP])
    tokens = tokenizer.encode(poem, add_special_tokens=False)
    
    # Dividir os tokens em partes de no máximo `max_tokens`
    parts = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    
    # Decodificar as partes de volta para texto
    return [tokenizer.decode(part) for part in parts]

# Função de pré-processamento dos textos
def preprocess_function(examples):
    try:
        # Dividir poemas longos em partes menores
        original_parts = split_poem(examples["original_poem"])
        translated_parts = split_poem(examples["translated_poem"])

        # Tokenizar cada parte
        inputs = tokenizer(original_parts, padding="max_length", truncation=True, max_length=512)
        targets = tokenizer(translated_parts, padding="max_length", truncation=True, max_length=512)
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
        output_dir="/home/ubuntu/finetuning/marianMT",
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

# Iniciar o treinamento
try:
    trainer.train()
except Exception as e:
    print(f"Erro durante o treinamento: {e}")
    exit(1)

# Salvar o modelo treinado
try:
    model.save_pretrained("/home/ubuntu/finetuning/marianMT_frances_ingles")
    tokenizer.save_pretrained("/home/ubuntu/finetuning/marianMT_frances_ingles")
    print("Fine-tuning finalizado e modelo salvo.")
except Exception as e:
    print(f"Erro ao salvar o modelo: {e}")
    exit(1)