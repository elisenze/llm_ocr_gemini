Create a virtual environment with Python >= 3.10

```
conda create --name myenv python=3.10 
conda activate myenv
pip install -r requirements.txt
```

If you don't have doppler installed, pdf2image will fail. Please install with 

```
sudo apt-get install poppler-utils
````
or 

```
brew install poppler

```


Run like this:

```
python main.py --pdf_file_path file_to_process.pdf
```


