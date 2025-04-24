import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

def baixar_e_extrair_dataset(dataset_id, destino='datasets'):
    nome_zip = f"{dataset_id.split('/')[-1]}.zip"
    arquivo_zip = os.path.join(destino, nome_zip)
    dataset_extraido = os.path.join(destino, 'annotations')

    if os.path.exists(dataset_extraido):
        print("📁 Dataset já está disponível. Pulando download.")
        return

    os.makedirs(destino, exist_ok=True)

    print("🔄 Baixando dataset do Kaggle...")
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(dataset_id, path=destino, unzip=False)
    print("✅ Download completo.")

    if not os.path.exists(arquivo_zip):
        raise FileNotFoundError(f"❌ Arquivo ZIP não encontrado: {arquivo_zip}")

    with zipfile.ZipFile(arquivo_zip, 'r') as zip_ref:
        zip_ref.extractall(destino)
        print("📦 Dataset extraído com sucesso.")

    os.remove(arquivo_zip)

