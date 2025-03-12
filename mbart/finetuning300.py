from datasets import load_dataset
from transformers import MBart50Tokenizer, MBartForConditionalGeneration
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

# Caminhos dos arquivos CSV
train_csv_path = "../modelos/poemas/train/poems_train.csv"
val_csv_path = "../modelos/poemas/validation/poems_validation.csv"

# Carregar dataset
dataset = load_dataset('csv', data_files={'train': train_csv_path, 'validation': val_csv_path})

# Carregar tokenizer
tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Função de pré-processamento
def preprocess_function(examples):
    src_langs = examples['src_lang']
    tgt_langs = examples['tgt_lang']
    original_poems = examples['original_poem']
    translated_poems = examples['translated_poem']

    input_ids_list, attention_mask_list, labels_list = [], [], []

    for src_lang, tgt_lang, original_poem, translated_poem in zip(src_langs, tgt_langs, original_poems, translated_poems):
        if isinstance(original_poem, str) and isinstance(translated_poem, str):  # Verifica se não é None
            tokenizer.src_lang = src_lang
            tokenizer.tgt_lang = tgt_lang

            inputs = tokenizer(original_poem, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
            labels = tokenizer(translated_poem, max_length=512, truncation=True, padding="max_length", return_tensors="pt")

            input_ids_list.append(inputs["input_ids"].squeeze(0))
            attention_mask_list.append(inputs["attention_mask"].squeeze(0))
            labels_list.append(labels["input_ids"].squeeze(0))

    # Retorna listas de tensores (com tamanho correto)
    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list)
    }

# Aplicar o pré-processamento sem remover exemplos
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# Definir formato PyTorch
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Carregar modelo e otimizador
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
optimizer = AdamW(model.parameters(), lr=2e-5)

# Criar DataLoaders
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=1)
val_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=1)

# Configurar dispositivo
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Loop de Treinamento
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1} - Training")):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        # Adiciona um print a cada 10 lotes para ver se está rodando
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch + 1} - Batch {batch_idx}/{len(train_dataloader)} - Loss: {loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss}")

    # Avaliação
    model.eval()
    total_eval_loss = 0
    for batch in tqdm(val_dataloader, desc=f"Epoch {epoch + 1} - Validation"):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_eval_loss += loss.item()

    avg_eval_loss = total_eval_loss / len(val_dataloader)
    print(f"Epoch {epoch + 1} - Validation Loss: {avg_eval_loss}")

# Salvar modelo treinado
model.save_pretrained("./fine-tuned-mbart-50-poems")
tokenizer.save_pretrained("./fine-tuned-mbart-50-poems")
