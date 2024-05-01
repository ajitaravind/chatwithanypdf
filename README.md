# Chat with any pdf

Create a RAG application using Llama3, Langchain and Groq to chat with any pdf document. Streamlit is used to build the front end. 

## Create a virtual environment,it saves lot of headaches later

```bash
conda create -name <nameofenvironment> <specify python version>
```
For example:
conda create --name myenv python=3.10.0

## Activate the virtual environment
```bash
conda activate myenv
```

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.txt.

```bash
pip install -r requirements.txt
```
## Run the application
```bash
python -m streamlit run chat.py
```
## License

[MIT](https://choosealicense.com/licenses/mit/)
