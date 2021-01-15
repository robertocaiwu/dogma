if [ -d ~/python_env/dogma ]; then
    echo "... python environment found."
else
    python3 -m venv ~/python_env/dogma
fi
source ~/python_env/dogma/bin/activate

 # Installation of python packages used by Kaldi wrapper
pip install --upgrade pip
pip install -r "requirements.txt"

deactivate