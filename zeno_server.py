from zeno import zeno
from zeno.api import model, distill, metric
from zeno.api import ModelReturn, MetricReturn, DistillReturn, ZenoOptions
import pandas as pd
from datasets import load_dataset
from transformers import pipeline
import tqdm

# ✅ Step 1: Load Dataset (Fix for NameError)
print("📂 Loading dataset...")
ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")  # Fetch dataset from Hugging Face
df = pd.DataFrame(ds['test']).head(500)  # Convert to DataFrame and limit to 500 samples

# ✅ Step 2: Map numeric labels to text labels
def label_map(x):
    return {0: "negative", 1: "neutral", 2: "positive"}.get(x, x)

df['label'] = df['label'].map(label_map)

# ✅ Step 3: Run Model Inference with RoBERTa
print("🔍 Running RoBERTa model inference...")
pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

results = []
for text in tqdm.tqdm(df['text'].to_list()):
    results.append(pipe(text))

df['roberta'] = [r[0]['label'] for r in results]
df['roberta_score'] = [r[0]['score'] for r in results]

# ✅ Step 4: Define Zeno Functions
@model
def load_model(model_name):
    def pred(df, ops: ZenoOptions):
        out = df[model_name]
        return ModelReturn(model_output=out)
    return pred

@distill
def label_match(df, ops: ZenoOptions):
    results = (df[ops.label_column] == df[ops.output_column]).to_list()
    return DistillReturn(distill_output=results)

@metric
def accuracy(df, ops: ZenoOptions):
    avg = df[ops.distill_columns["label_match"]].mean()
    return MetricReturn(metric=avg)

# ✅ Step 5: Start the Zeno Server
if __name__ == "__main__":
    print("🚀 Starting Zeno server on http://localhost:8231 ...")
    zeno({
        "metadata": df,
        "view": "text-classification",
        "id_column": "text",
        "label_column": "label",
        "functions": [load_model, label_match, accuracy],
        "models": ["roberta"],
        "port": 8231,
        "host": "127.0.0.1"
    })
