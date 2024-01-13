<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Allenamento distribuito con 🤗 Accelerate

La parallelizzazione è emersa come strategia per allenare modelli sempre più grandi su hardware limitato e accelerarne la velocità di allenamento di diversi ordini di magnitudine. In Hugging Face, abbiamo creato la libreria [🤗 Accelerate](https://huggingface.co/docs/accelerate) per aiutarti ad allenare in modo semplice un modello 🤗 Transformers su qualsiasi tipo di configurazione distribuita, sia che si tratti di più GPU su una sola macchina o di più GPU su più macchine. In questo tutorial, imparerai come personalizzare il training loop nativo di PyTorch per consentire l'addestramento in un ambiente distribuito.

## Configurazione

Inizia installando 🤗 Accelerate:

```bash
pip install accelerate
```

Poi importa e crea un oggetto [`Accelerator`](https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator). `Accelerator` rileverà automaticamente il tuo setup distribuito e inizializzerà tutte le componenti necessarie per l'allenamento. Non dovrai allocare esplicitamente il tuo modello su un device.

```py
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
```

## Preparati ad accelerare

Il prossimo passo è quello di passare tutti gli oggetti rilevanti per l'allenamento al metodo [`prepare`](https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.prepare). Questo include i tuoi DataLoaders per l'allenamento e per la valutazione, un modello e un ottimizzatore:

```py
>>> train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
...     train_dataloader, eval_dataloader, model, optimizer
... )
```

## Backward

Infine, sostituisci il tipico metodo `loss.backward()` nel tuo loop di allenamento con il metodo [`backward`](https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.backward) di 🤗 Accelerate:

```py
>>> for epoch in range(num_epochs):
...     for batch in train_dataloader:
...         outputs = model(**batch)
...         loss = outputs.loss
...         accelerator.backward(loss)

...         optimizer.step()
...         lr_scheduler.step()
...         optimizer.zero_grad()
...         progress_bar.update(1)
```

Come puoi vedere nel seguente codice, hai solo bisogno di aggiungere quattro righe in più di codice al tuo training loop per abilitare l'allenamento distribuito!

```diff
+ from accelerate import Accelerator
  from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

+ accelerator = Accelerator()

  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
  optimizer = AdamW(model.parameters(), lr=3e-5)

- device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
- model.to(device)

+ train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
+     train_dataloader, eval_dataloader, model, optimizer
+ )

  num_epochs = 3
  num_training_steps = num_epochs * len(train_dataloader)
  lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps
  )

  progress_bar = tqdm(range(num_training_steps))

  model.train()
  for epoch in range(num_epochs):
      for batch in train_dataloader:
-         batch = {k: v.to(device) for k, v in batch.items()}
          outputs = model(**batch)
          loss = outputs.loss
-         loss.backward()
+         accelerator.backward(loss)

          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          progress_bar.update(1)
```

## Allenamento

Una volta che hai aggiunto le righe di codice rilevanti, lancia il tuo allenamento in uno script o in un notebook come Colaboratory.

### Allenamento con uno script

Se stai eseguendo il tuo allenamento da uno script, esegui il comando seguente per creare e salvare un file di configurazione:

```bash
accelerate config
```

Poi lancia il tuo allenamento con:

```bash
accelerate launch train.py
```

### Allenamento con un notebook

La libreria 🤗 Accelerate può anche essere utilizzata in un notebook se stai pianificando di utilizzare le TPU di Colaboratory. Inserisci tutto il codice legato all'allenamento in una funzione, e passala al `notebook_launcher`:

```py
>>> from accelerate import notebook_launcher

>>> notebook_launcher(training_function)
```

Per maggiori informazioni relative a 🤗 Accelerate e le sue numerose funzionalità, fai riferimento alla [documentazione](https://huggingface.co/docs/accelerate).