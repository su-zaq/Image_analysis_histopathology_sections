# Image_analysis_histopathology_sections

## 環境構築手順

### 1. 仮想環境の作成と有効化

```bash
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate
bash```

以下のコマンドで必要なライブラリをインストールします。
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/Kotetsu0000/VitLib.git
pip install git+https://github.com/Kotetsu0000/VitLib_PyTorch.git
pip install pandas
bash```
