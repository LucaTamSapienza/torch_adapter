# torch_adapter

The purpose of this project is to provide an adapter for **PyTorch**, entirely using the **OpenVINO API**. For now, only the preprocessing and inference steps are available.

## Requirments:
- python >= 3.9
- venv
---
> [!IMPORTANT]
> If you encounter any issues or believe some dependencies are missing, please don't hesitate to contact me. Also, ensure to double-check each command to confirm that everything is functioning as expected.

## Installation
You can either choose to install `torch_adapter` from pip (preferred) or build it from source.

> I suggest creating a virtual environment before installing torch_adapter.

```python
pip install torch_adapter
```

Or, if you want to work on the code, just follow these steps:

```sh
git clone git@github.com:LucaTamSapienza/torch_adapter.git
cd torch_adapter
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You may need to export your PYTHONPATH to point to the torch_adapter directory:

```
export PYTHONPATH=$PYTHONPATH:path_to_torch_adapter
```

Now you can check if everything works by running `pytest tests/`
