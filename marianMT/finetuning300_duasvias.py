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
train_csv_fr = "../poemas/poemas300/train/frances_portugues_train.csv"
val_csv_fr = "../poemas/poemas300/validation/frances_portugues_validation.csv"
intermediate_csv_path = "../poemas/poemas300/train/frances_portugues_train_intermediario.csv"

# Função para carregar datasets do Hugging Face
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

# Função de pré-processamento dos textos
def preprocess_function(examples, tokenizer):
    try:
        inputs = tokenizer(examples["original_poem"], padding="max_length", truncation=True, max_length=512)
        targets = tokenizer(examples["translated_poem"], padding="max_length", truncation=True, max_length=512)
        inputs["labels"] = targets["input_ids"]
        return inputs
    except Exception as e:
        print(f"Erro durante o pré-processamento: {e}")
        raise

# Função para treinar um modelo MarianMT
def train_model(model_name, train_csv, val_csv, output_dir):
    try:
        train_dataset = load_data(train_csv)
        val_dataset = load_data(val_csv)

        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(device)

        # Aplicar pré-processamento
        train_dataset = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
        val_dataset = val_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
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

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        )

        trainer.train()

        # Salvar modelo treinado
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Fine-tuning finalizado e modelo salvo em {output_dir}")

        return model, tokenizer

    except Exception as e:
        print(f"Erro durante o treinamento do modelo {model_name}: {e}")
        exit(1)

# Função para traduzir textos
def translate_texts(texts, model, tokenizer, target_lang=None):
    if target_lang:
        texts = [f">>{target_lang}<< {text}" for text in texts]  # Adiciona o token de idioma
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    translated = model.generate(**inputs)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

# --------- ETAPA 1: Treinamento Francês → Inglês ---------
print("\nIniciando o fine-tuning Francês → Inglês...")
model_fr_en, tokenizer_fr_en = train_model(
    model_name="Helsinki-NLP/opus-mt-fr-en", 
    train_csv=train_csv_fr, 
    val_csv=val_csv_fr, 
    output_dir="/home/ubuntu/finetuning/marianMT/marianMT_frances_portugues_intermediario"
)

# Gerar dataset intermediário (Francês → Inglês)
df_fr = pd.read_csv(train_csv_fr)
print("Gerando traduções intermediárias do Francês para o Inglês...")
english_translations = translate_texts(df_fr["original_poem"].tolist(), model_fr_en, tokenizer_fr_en)

# Criar novo dataset para a próxima etapa (Inglês → Português)
df_intermediate = pd.DataFrame({
    "original_poem": english_translations,
    "translated_poem": df_fr["translated_poem"]  # A referência humana já está em português
})

# Salvar dataset intermediário
df_intermediate.to_csv(intermediate_csv_path, index=False)

# --------- ETAPA 2: Treinamento Inglês → Português ---------
print("\nIniciando o fine-tuning Inglês → Português...")
model_en_pt, tokenizer_en_pt = train_model(
    model_name="Helsinki-NLP/opus-mt-en-ROMANCE",  # Modelo correto para en → pt
    train_csv=intermediate_csv_path,
    val_csv=val_csv_fr,  # Reutiliza o dataset original
    output_dir="/home/ubuntu/finetuning/marianMT/marianMT_frances_portugues"
)

# --------- ETAPA 3: Teste de Tradução Francês → Português ---------
print("\nCarregando modelo final para testes...")
final_model_path = "/home/ubuntu/finetuning/marianMT/marianMT_frances_portugues"
final_tokenizer = MarianTokenizer.from_pretrained(final_model_path)
final_model = MarianMTModel.from_pretrained(final_model_path).to(device)

# Função para traduzir poemas diretamente do Francês para o Português
def translate_poem_fr_to_pt(poem_text):
    poem_text = f">>pt<< {poem_text}"  # Indica que a saída deve ser em português
    inputs = final_tokenizer(poem_text, return_tensors="pt", padding=True, truncation=True).to(device)
    translated = final_model.generate(**inputs)
    return final_tokenizer.decode(translated[0], skip_special_tokens=True)

# Calcular tempo total de execução
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tempo total de execução: {elapsed_time:.2f} segundos")
