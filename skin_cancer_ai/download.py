import kagglehub

# Download latest version
path = kagglehub.dataset_download("eliocordeiropereira/skin-cancer-the-ham10000-dataset")

print("Path to dataset files:", path)
# Вот тут то что скачается должны в папку data переместить. Папку создать в этой директории
# C:\Users\username\.cache\kagglehub\datasets\eliocordeiropereira\skin-cancer-the-ham10000-dataset\versions\1
# Вот оттуда все вставляете в data папку. Тогда примерно будет skin_cancer_ai/data/*
