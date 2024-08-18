{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"name":"python","version":"3.10.13","mimetype":"text/x-python","codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3","nbconvert_exporter":"python","file_extension":".py"},"kaggle":{"accelerator":"none","dataSources":[],"dockerImageVersionId":30746,"isInternetEnabled":true,"language":"python","sourceType":"notebook","isGpuEnabled":false}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"root='/kaggle/working/'\nimport sys\nmodule_path = root+'thesis/src/EDA'\nsys.path.append(module_path)\nimport os\nimport numpy as np\nimport pandas as pd\nfrom tqdm import tqdm\nimport seaborn as sns\nfrom scipy import stats\nimport random\nfrom scipy.stats import chi2\nimport statsmodels.api as sm\nimport matplotlib.pyplot as plt\nfrom statsmodels.graphics.gofplots import qqplot\nimport pandas as pd\n!pip install fastparquet\n!pip install gdown","metadata":{"execution":{"iopub.status.busy":"2024-08-18T15:54:53.574051Z","iopub.execute_input":"2024-08-18T15:54:53.574485Z","iopub.status.idle":"2024-08-18T15:55:29.133519Z","shell.execute_reply.started":"2024-08-18T15:54:53.574448Z","shell.execute_reply":"2024-08-18T15:55:29.132104Z"},"trusted":true},"execution_count":1,"outputs":[{"name":"stdout","text":"Requirement already satisfied: fastparquet in /opt/conda/lib/python3.10/site-packages (2024.5.0)\nRequirement already satisfied: pandas>=1.5.0 in /opt/conda/lib/python3.10/site-packages (from fastparquet) (2.2.2)\nRequirement already satisfied: numpy in /opt/conda/lib/python3.10/site-packages (from fastparquet) (1.26.4)\nRequirement already satisfied: cramjam>=2.3 in /opt/conda/lib/python3.10/site-packages (from fastparquet) (2.8.3)\nRequirement already satisfied: fsspec in /opt/conda/lib/python3.10/site-packages (from fastparquet) (2024.5.0)\nRequirement already satisfied: packaging in /opt/conda/lib/python3.10/site-packages (from fastparquet) (21.3)\nRequirement already satisfied: python-dateutil>=2.8.2 in /opt/conda/lib/python3.10/site-packages (from pandas>=1.5.0->fastparquet) (2.9.0.post0)\nRequirement already satisfied: pytz>=2020.1 in /opt/conda/lib/python3.10/site-packages (from pandas>=1.5.0->fastparquet) (2023.3.post1)\nRequirement already satisfied: tzdata>=2022.7 in /opt/conda/lib/python3.10/site-packages (from pandas>=1.5.0->fastparquet) (2023.4)\nRequirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /opt/conda/lib/python3.10/site-packages (from packaging->fastparquet) (3.1.1)\nRequirement already satisfied: six>=1.5 in /opt/conda/lib/python3.10/site-packages (from python-dateutil>=2.8.2->pandas>=1.5.0->fastparquet) (1.16.0)\nRequirement already satisfied: gdown in /opt/conda/lib/python3.10/site-packages (5.2.0)\nRequirement already satisfied: beautifulsoup4 in /opt/conda/lib/python3.10/site-packages (from gdown) (4.12.2)\nRequirement already satisfied: filelock in /opt/conda/lib/python3.10/site-packages (from gdown) (3.13.1)\nRequirement already satisfied: requests[socks] in /opt/conda/lib/python3.10/site-packages (from gdown) (2.32.3)\nRequirement already satisfied: tqdm in /opt/conda/lib/python3.10/site-packages (from gdown) (4.66.4)\nRequirement already satisfied: soupsieve>1.2 in /opt/conda/lib/python3.10/site-packages (from beautifulsoup4->gdown) (2.5)\nRequirement already satisfied: charset-normalizer<4,>=2 in /opt/conda/lib/python3.10/site-packages (from requests[socks]->gdown) (3.3.2)\nRequirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.10/site-packages (from requests[socks]->gdown) (3.6)\nRequirement already satisfied: urllib3<3,>=1.21.1 in /opt/conda/lib/python3.10/site-packages (from requests[socks]->gdown) (1.26.18)\nRequirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.10/site-packages (from requests[socks]->gdown) (2024.7.4)\nRequirement already satisfied: PySocks!=1.5.7,>=1.5.6 in /opt/conda/lib/python3.10/site-packages (from requests[socks]->gdown) (1.7.1)\n","output_type":"stream"}]},{"cell_type":"code","source":"import gdown\nurl='https://drive.google.com/uc?id=1Y2lGWkcgo_IWHdWFh_Qcn0K74D_xQhvB'\noutput='facebook_news.csv'\ngdown.download(url,output,quiet=False)\n","metadata":{"execution":{"iopub.status.busy":"2024-08-18T15:55:29.136487Z","iopub.execute_input":"2024-08-18T15:55:29.137903Z","iopub.status.idle":"2024-08-18T15:55:43.374581Z","shell.execute_reply.started":"2024-08-18T15:55:29.137857Z","shell.execute_reply":"2024-08-18T15:55:43.373379Z"},"trusted":true},"execution_count":2,"outputs":[{"name":"stderr","text":"Downloading...\nFrom (original): https://drive.google.com/uc?id=1Y2lGWkcgo_IWHdWFh_Qcn0K74D_xQhvB\nFrom (redirected): https://drive.google.com/uc?id=1Y2lGWkcgo_IWHdWFh_Qcn0K74D_xQhvB&confirm=t&uuid=2dd1f62c-17fa-4bf3-a046-416b34b75b8f\nTo: /kaggle/working/facebook_news.csv\n100%|██████████| 1.84G/1.84G [00:08<00:00, 219MB/s]\n","output_type":"stream"},{"execution_count":2,"output_type":"execute_result","data":{"text/plain":"'facebook_news.csv'"},"metadata":{}}]},{"cell_type":"code","source":"url='https://drive.google.com/uc?id=1QepHehlhqP-jtOcshFzajqxj1DpJtoj7'\n\noutput='reddit.parquet'\n\ngdown.download(url,output,quiet=False)","metadata":{"execution":{"iopub.status.busy":"2024-08-18T16:17:56.913326Z","iopub.execute_input":"2024-08-18T16:17:56.913875Z","iopub.status.idle":"2024-08-18T16:18:02.717530Z","shell.execute_reply.started":"2024-08-18T16:17:56.913836Z","shell.execute_reply":"2024-08-18T16:18:02.716329Z"},"trusted":true},"execution_count":16,"outputs":[{"name":"stderr","text":"Downloading...\nFrom (original): https://drive.google.com/uc?id=1QepHehlhqP-jtOcshFzajqxj1DpJtoj7\nFrom (redirected): https://drive.google.com/uc?id=1QepHehlhqP-jtOcshFzajqxj1DpJtoj7&confirm=t&uuid=e00ee6bd-8cec-4064-befd-94a82b275a60\nTo: /kaggle/working/reddit.parquet\n100%|██████████| 334M/334M [00:01<00:00, 173MB/s]  \n","output_type":"stream"},{"execution_count":16,"output_type":"execute_result","data":{"text/plain":"'reddit.parquet'"},"metadata":{}}]},{"cell_type":"markdown","source":"## Facebook","metadata":{}},{"cell_type":"code","source":"source=pd.read_csv('/kaggle/working/facebook_news.csv')\ndata = source.dropna(subset=['toxicity_score'])\ndata=data.groupby(['user_id', 'post_id'])['toxicity_score'].mean().reset_index()\nimport pandas as pd\nimport numpy as np\n\n# Supponiamo che il tuo dataset sia già caricato in un DataFrame chiamato facebook_news\n\ndef gini_coefficient(x):\n    # Sort the values\n    x = np.sort(x)\n    n = len(x)\n    if n == 0:\n        return 0\n    \n    # Compute the cumulative sum of the sorted values\n    cumx = np.cumsum(x)\n    \n    # Compute the Gini coefficient using the cumulative sum\n    gini = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n\n    \n    return gini\n\n# Aggrega i dati per conversation_id\ngrouped = data.groupby('post_id')['toxicity_score']\n\n# Calcola l'indice di Gini per ogni gruppo\ngini_results = grouped.apply(lambda x: gini_coefficient(x))\n# Trasforma il risultato in un DataFrame\ngini_df = gini_results.reset_index(name='gini_index')\nprint(gini_df)\n# Trasforma il risultato in un DataFrame\ngini_df = gini_results.reset_index(name='gini_index')\n# Plot della distribuzione dell'indice di Gini\ngini_df=gini_df[gini_df['gini_index']>0]\nplt.figure(figsize=(10, 6))\nsns.histplot(gini_df['gini_index'], bins=30, kde=True)\nplt.title('Distribuzione dell\\'Indice di Gini per ciascuna Conversazione')\nplt.xlabel('Indice di Gini')\nplt.ylabel('Frequenza')\nplt.xlim(0,1)\nplt.show()","metadata":{"execution":{"iopub.status.busy":"2024-08-18T16:26:17.253604Z","iopub.execute_input":"2024-08-18T16:26:17.255175Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"gini_df['gini_index'].mean()","metadata":{"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":"## Reddit","metadata":{}},{"cell_type":"code","source":"!pip install fastparquet\nsource=pd.read_parquet('/kaggle/working/reddit.parquet' ,engine='fastparquet')\ndata = source.dropna(subset=['toxicity_score'])\n\n\ndata=data.groupby(['user_id', 'post_id'])['toxicity_score'].mean().reset_index()\n\nimport pandas as pd\nimport numpy as np\n\n# Supponiamo che il tuo dataset sia già caricato in un DataFrame chiamato facebook_news\n\ndef gini_coefficient(x):\n    # Sort the values\n    x = np.sort(x)\n    n = len(x)\n    if n == 0:\n        return 0\n    \n    # Compute the cumulative sum of the sorted values\n    cumx = np.cumsum(x)\n    \n    # Compute the Gini coefficient using the cumulative sum\n    gini = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n\n    \n    return gini\n\n# Aggrega i dati per conversation_id\ngrouped = data.groupby('post_id')['toxicity_score']\n\n# Calcola l'indice di Gini per ogni gruppo\ngini_results = grouped.apply(lambda x: gini_coefficient(x))\n# Trasforma il risultato in un DataFrame\ngini_df = gini_results.reset_index(name='gini_index')\nprint(gini_df)\n# Trasforma il risultato in un DataFrame\ngini_df = gini_results.reset_index(name='gini_index')\n# Plot della distribuzione dell'indice di Gini\ngini_df=gini_df[gini_df['gini_index']>0]\nplt.figure(figsize=(10, 6))\nsns.histplot(gini_df['gini_index'], bins=30, kde=True)\nplt.title('Distribuzione dell\\'Indice di Gini per ciascuna Conversazione')\nplt.xlabel('Indice di Gini')\nplt.ylabel('Frequenza')\nplt.xlim(0,1)\nplt.show()","metadata":{"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"gini_df['gini_index'].mean()","metadata":{"execution":{"iopub.status.busy":"2024-08-18T16:18:36.215848Z","iopub.execute_input":"2024-08-18T16:18:36.216310Z","iopub.status.idle":"2024-08-18T16:18:36.226661Z","shell.execute_reply.started":"2024-08-18T16:18:36.216273Z","shell.execute_reply":"2024-08-18T16:18:36.225394Z"},"trusted":true},"execution_count":18,"outputs":[{"execution_count":18,"output_type":"execute_result","data":{"text/plain":"0.39841407514446375"},"metadata":{}}]}]}