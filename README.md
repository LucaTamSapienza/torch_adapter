# torch_adapter

**Requirments:**
- python >= 3.9
- venv

> [!IMPORTANT]
> If you encounter any issues or believe some dependencies are missing, please don't hesitate to contact me. Also, ensure to double-check each command to confirm that everything is functioning as expected.
```sh
git clone git@github.com:LucaTamSapienza/torch_adapter.git
cd torch_adapter
git checkout pre-release
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You may need to export your PYTHONPATH to point to the torch_adapter directory:


```
export PYTHONPATH=$PYTHONPATH:path_to_torch_adapter
```

> [!NOTE]
> Please replace path_to_torch_adapter with the actual path to the torch_adapter directory on your system.

Now you can check if everything works by running `pytest tests/`
