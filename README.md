# Song classifier

This project uses python with scikit-learn's random Forest Classifier to classify a song in three possible genres: Metal, Disco and Reggaeton.

## Install

Once cloned the repository in order to install the project run:

```console
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt

mkdir ./res/music/wav/
python setup.py
```


## Usage

Look for a youtube song and copy its link, then run (the link has to be between quotemarks):

```console
python classify.py {link}
```


## Re-train

If you want to train once again the model run (Warning: this process may take a while):

```console
python trainer.py
```
