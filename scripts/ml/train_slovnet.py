import sys
from pathlib import Path

%cd /slovnet/scripts/05_ner
%run main.py
CUSTOM_TEXTS = ""


TAGS = ["PERSON", "ADDRESS", "EMAIL", "ID", "PHONE"]
DEVICE = "cpu"

LAYERS_NUM = 4
LAYER_DIM = 128
LAYER_DIMS = [
    LAYER_DIM * 2 ** _
    for _ in reversed(range(LAYERS_NUM))
]

LR = 0.0005
LR_GAMMA = 0.95
EPOCHS = 15

BATCH_SIZE = 8
SEQ_LEN = 256
SHUFFLE_SIZE = 1000


navec = Navec.load(NAVEC)

words_vocab = Vocab(navec.vocab.words)
shapes_vocab = Vocab([PAD] + SHAPES)
tags_vocab = BIOTagsVocab(TAGS)

torch.manual_seed(SEED)
seed(SEED)

word = NavecEmbedding(navec)

shape = Embedding(
    vocab_size=len(shapes_vocab),
    dim=SHAPE_DIM,
    pad_id=shapes_vocab.pad_id,
)

emb = TagEmbedding(word, shape)

encoder = TagEncoder(
    input_dim=emb.dim,
    layer_dims=LAYER_DIMS,
    kernel_size=KERNEL_SIZE,
)

ner = NERHead(encoder.dim, len(tags_vocab))
model = NER(emb, encoder, ner).to(DEVICE)

lines = load_gz_lines(CUSTOM_TEXTS)
items = parse_jl(lines)

markups = (SpanMarkup.from_json(_) for _ in items)
markups = (_.to_bio(list(tokenize(_.text))) for _ in markups)

encode = TagTrainEncoder(
    words_vocab,
    shapes_vocab,
    tags_vocab,
    seq_len=SEQ_LEN,
    batch_size=BATCH_SIZE,
    shuffle_size=SHUFFLE_SIZE,
)

batches = list(encode(markups))
batches = [_.to(DEVICE) for _ in batches]

size = max(1, int(len(batches) * 0.15))

batches = {
    TEST: batches[:size],
    TRAIN: batches[size:],
}


optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, LR_GAMMA)

meters = {
    TRAIN: NERScoreMeter(),
    TEST: NERScoreMeter(),
}

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    model.train()
    for batch in batches[TRAIN]:
        optimizer.zero_grad()
        batch = process_batch(model, ner.crf, batch)
        batch.loss.backward()
        optimizer.step()
        meters[TRAIN].add(NERBatchScore(batch.loss))

    print("Train:")
    meters[TRAIN].write(LogBoard())
    meters[TRAIN].reset()

    model.eval()
    with torch.no_grad():
        for batch in batches[TEST]:
            batch = process_batch(model, ner.crf, batch)
            batch.pred = ner.crf.decode(batch.pred)
            meters[TEST].add(score_ner_batch(batch, tags_vocab))

    print("Valid:")
    meters[TEST].write(LogBoard())
    meters[TEST].reset()

    scheduler.step()

model.emb.shape.dump(MODEL_SHAPE)
model.encoder.dump(MODEL_ENCODER)
ner.dump(MODEL_NER)

!ls model
