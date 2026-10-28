# What is retrieval-augmented generation?

Retrieval-augmented generation, usually shortened to RAG, is a pattern
for making a language model answer from your own documents instead of
only from what it memorised during training.

It works in two steps. First, retrieval: the question is embedded into
a vector, and the system finds the document chunks whose embeddings are
closest to it, usually by cosine similarity. Second, generation: those
chunks are pasted into the prompt as context, and the model is told to
answer only from that context.

The main benefit is grounding. A plain language model can confidently
invent facts. A RAG system can cite the exact chunk its answer came
from, and it can honestly say it does not know when the documents are
silent. The trade-off is that the answer is only as good as the
retrieval step: if the right chunk is not retrieved, the model cannot
use it, no matter how good the prompt is.

Good chunking matters more than people expect. Chunks that are too long
dilute the signal, because one relevant sentence hides among many
irrelevant ones. Chunks that are too short lose the surrounding
context. Overlapping windows are a simple fix for facts that fall on a
boundary between two chunks.
