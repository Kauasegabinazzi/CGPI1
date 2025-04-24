# Programa em Python para Detecção de Máscaras Faciais em Humanos

## Nomes

- Kauã Segabinazzi
- Leonardo Rossi Quines
- Henrique Correa
- Jonathan D. Hartmann

## Dataset: (https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

## Instalação e Execução:

O projeto *raw* pode ser instalado no *GitHub* no formato `.zip` ou clonado via `git clone <URL do repositório>` na pasta que você deseja hospedar o repositório (o qual pode ser acessado no sistema com `cd <nome_do_repositorio>`.

Após a instalação é preciso configurar o ambiente, isso pode ser feito com o comando `python -m venv venv`, que gera um ambiente virtual para que as dependências sejam instaladas na versão correta e apenas no ambiente virutal. Para ativar o ambiente virtual no *Windows*, use o comando `venv\Scripts\activate`.

Posteriormente deve-se executar o comando `pip install -r requirements.txt`, que contém todas as dependências do projeto.

Seguidos esses passos, é possível começar a editar o repositório e o continuar.

O projeto realiza automaticamente o download e extração do dataset via API do Kaggle. Para isso funcionar corretamente, siga os passos abaixo:

Acesse: https://www.kaggle.com/account

Role até a seção "API" e clique em "Create New API Token"
Isso vai gerar e baixar um arquivo chamado kaggle.json.

Coloque o kaggle.json dentro de uma pasta chamada .kaggle no seu diretório de usuário: `C:\Users\<seu-usuario>\.kaggle\kaggle.json`

Após configurar o kaggle.json, basta rodar: `python coletor_dataset.py`


