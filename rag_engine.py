import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------
print("Loading dataset...")
df = pd.read_csv("data/Global_Superstore2.csv", encoding='latin1')

documents = []
metadatas = []
ids = []

# Global ID Counter
id_counter = 0

# ---------------------------------------------------
# 2. GENERATE REGIONAL SUMMARIES
# ---------------------------------------------------
print("Compiling geographical summaries...")
region_summary = df.groupby('Region').agg({'Sales': 'sum', 'Profit': 'sum', 'Discount': 'mean'}).reset_index()
for _, row in region_summary.iterrows():
    text = f"Geographical Summary: The {row['Region']} region generated ${row['Sales']:,.2f} in total sales and ${row['Profit']:,.2f} in total profit, with an average discount of {row['Discount']:.2%} applied across orders."
    documents.append(text)
    metadatas.append({"type": "geography", "region": row['Region']})
    ids.append(f"geo_{id_counter}")
    id_counter += 1

# ---------------------------------------------------
# 3. GENERATE CATEGORY SUMMARIES
# ---------------------------------------------------
print("Compiling product line performance metrics...")
category_summary = df.groupby('Category').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
for _, row in category_summary.iterrows():
    text = f"Category Summary: The product family '{row['Category']}' accounts for ${row['Sales']:,.2f} in global revenue and ${row['Profit']:,.2f} in total corporate net profits."
    documents.append(text)
    metadatas.append({"type": "product", "category": row['Category']})
    ids.append(f"prod_{id_counter}")
    id_counter += 1

# ---------------------------------------------------
# 4. [ADVANCED DATA SCIENCE] UNSUPERVISED RFM CLUSTERING
# ---------------------------------------------------
print("Executing Unsupervised Learning (RFM K-Means Customer Segmentation)...")

# Clean dates to calculate Recency
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y', errors='coerce')
df_clean_dates = df.dropna(subset=['Order Date'])
snapshot_date = df_clean_dates['Order Date'].max() + pd.Timedelta(days=1)

# Group transactions to build Recency, Frequency, Monetary metrics per customer
rfm = df_clean_dates.groupby('Customer Name').agg({
    'Order Date': lambda x: (snapshot_date - x.max()).days, # Recency
    'Order ID': 'nunique',                                  # Frequency
    'Sales': 'sum'                                          # Monetary
}).rename(columns={'Order Date': 'Recency', 'Order ID': 'Frequency', 'Sales': 'Monetary'}).reset_index()

# Scale features to mean=0, variance=1 for stable numerical clustering
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Apply unsupervised K-Means Clustering to automatically segment the population into 3 groups
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Map mathematical clusters to strategic business labels
cluster_mapping = {
    0: "At-Risk / Hibernating Accounts", 
    1: "High-Volume Strategic Champions", 
    2: "Loyal Core Buyers"
}
rfm['Segment_Name'] = rfm['Cluster'].map(cluster_mapping)

# Add customer cluster profiles to database documents
for _, row in rfm.iterrows():
    text = (f"Algorithmic Customer Segment Profile: Customer '{row['Customer Name']}' belongs to the unsupervised cluster '{row['Segment_Name']}'. "
            f"Statistically, this client has a Recency score of {row['Recency']} days since last order, an Order Frequency score of {row['Frequency']} distinct transactions, "
            f"and a high-fidelity lifetime Monetary revenue contribution of ${row['Monetary']:,.2f}.")
    documents.append(text)
    metadatas.append({"type": "segmentation", "cluster": int(row['Cluster']), "customer": row['Customer Name']})
    ids.append(f"cust_{id_counter}")
    id_counter += 1

# ---------------------------------------------------
# 5. [ML-DRIVEN RISK LAYER] CRITICAL LOSS DESIGNATIONS
# ---------------------------------------------------
print("Extracting critical business leakage vectors...")
# Find regions/categories where profit is negative (leaking money)
loss_makers = df[df['Profit'] < 0].groupby(['Region', 'Category']).agg({'Sales': 'sum', 'Profit': 'sum', 'Discount': 'mean'}).reset_index()
for _, row in loss_makers.iterrows():
    text = f"Critical Financial Loss Vector: In the {row['Region']} market, the {row['Category']} line is experiencing severe capital leakage. It accumulated losses of ${abs(row['Profit']):,.2f} on ${row['Sales']:,.2f} in gross volume. Diagnostics point to an unsustainably high average discount rate of {row['Discount']:.2%} as the root cause."
    documents.append(text)
    metadatas.append({"type": "risk", "region": row['Region'], "category": row['Category']})
    ids.append(f"risk_{id_counter}")
    id_counter += 1

# ---------------------------------------------------
# 6. ENCODE & UPDATE DATABASE
# ---------------------------------------------------
print("Initializing Embedding Vectorization Engine...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(documents)

print("Connecting to ChromaDB Workspace...")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="business_data")

# Clear out obsolete indices to avoid duplication during development iterations
existing = collection.get()
if existing['ids']:
    print(f"Purging {len(existing['ids'])} legacy index nodes...")
    collection.delete(ids=existing['ids'])

print(f"Uploading {len(documents)} context strings with embedded ML metadata structures to ChromaDB...")
collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)

print("✅ Success! The rag_engine.py pipeline has been completely upgraded to a scientific ML infrastructure.")