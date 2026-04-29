# Image_analysis_histopathology_sections

薄い病理標本画像から、**細胞膜**および**細胞核**の領域を抽出するための学習・推論・評価用コードです。入力は明視野（bf）・暗視野（df）・位相差（ph）の各撮像画像を組み合わせ、U-Net / U-Net++（VitLib_PyTorch）でセグメンテーションを行います。

## リポジトリ構成（主要ファイル）

| ファイル・ディレクトリ | 役割 |
|------------------------|------|
| `experiment.py` | 学習のメイン。データ拡張・交差検証ループ・推論画像保存・外部フォルダ一括推論・モデル保存など |
| `evaluation.py` | 学習で出力した推論結果に対し、VitLib の評価関数で疎探索→密探索→集計（CSV） |
| `infer.py` | 学習済みチェックポイントを指定し、任意フォルダ内の画像だけを推論（オフライン用） |
| `Dataset/` | `Dataset_experiment_single`（膜・核単独）、`both`（同時学習）、`plus`（膜+／核+で相手マスクを条件入力） |

補助モジュール `send_info_discord.py` は `.gitignore` に含まれており、通知が必要な場合はローカルで用意してください（無いと import エラーになります）。

## データ分割の考え方

標本フォルダを **N 個**に分けたとき、**N≥3** なら **学習 : 検証 : テスト = (N−2) : 1 : 1**、**N=2** なら **学習 : テスト = 1 : 1** です（`experiment.py` 先頭コメント参照）。撮像法のオンオフ組み合わせ（`use_list_length` と `blend` により定義）ごとに上記ループが回ります。

## 元データのフォルダ構成（`img_path`）

学習用のルート（既定は `./Data/master_exp_data`）直下に、**同一標本ずつ**フォルダを置き、その中に**フィールド画像単位**のサブフォルダを並べます。各サブフォルダには次を置きます。

- `x/` … 撮像画像（**`bf.png`**, **`df.png`**, **`ph.png`**）
- `y_membrane/` … 膜用。細線化ラベル **`ans_thin.png`**（学習用 `ans.png` 等は実行時に生成）
- `y_nuclear/` … 核用 **`ans.png`** など（Don't care 用の派生画像も学習準備で生成）

標本フォルダは **2 個以上**必要です。

## 環境構築手順

### 1. 仮想環境の作成と有効化

```bash
python -m venv venv
```

- Windows: `venv\Scripts\activate`
- Linux / macOS: `source venv/bin/activate`

### 2. 依存パッケージのインストール

CUDA 対応 PyTorch の例（README 作成時点の指定に合わせています。環境に合わせて [PyTorch の案内](https://pytorch.org/) を参照してください）。

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/Kotetsu0000/VitLib.git
pip install git+https://github.com/Kotetsu0000/VitLib_PyTorch.git
pip install pandas opencv-python-headless pillow tqdm
```

**注意:** `experiment.py` は起動時に **CUDA 必須**（`assert torch.cuda.is_available()`）です。CPU のみの環境ではそのまま実行できません。

## 学習の実行（`experiment.py`）

- パラメータはクラス `Extraction` の引数、または **INI 形式の設定ファイル**（`--config` / `-c`）で指定します。
- 初回（`start_num == 0`）は `default_path` 下にログ等を作成します。既に `log/exp.log` があると再実行で例外になるため、続きから回す場合は `start_num` を調整するか、出力先を空にしてください（`ignore_error` も関連）。
- 膜・核・同時学習・条件付き（`membrane+` / `nuclear+`）など **`experiment_subject`** により分岐します。
- **epoch 20** でモデルを `result/.../model/expXXXX_testYY_epoch20.pth` 形式で保存。各 epoch 終了後、検証・テスト画像の保存、および **`./test/x` 直下の画像**に対する外部推論（`result/.../external_pred/...`）が走ります（フォルダが無い・画像が無い場合はスキップログ）。

```bash
python experiment.py
python experiment.py -c your_config.ini
```

## 評価（`evaluation.py`）

`experiment.py` の出力先（`default_path`）と、正解のある元データルート（`img_path`）を設定し、**疎探索（閾値・小領域削除の幅広い探索）→ 密探索（近傍を細かく）→ バリデーション／テストの集計**の順で CSV を生成します。設定は `--config` の `[EXPERIMENT_PATH]` `[EXPERIMENT_PARAM]` `[EVALIATION_PARAM]` を参照してください。

```bash
python evaluation.py -c your_config.ini
```

## 推論のみ（`infer.py`）

学習に使った **入力チャンネル数・ネットワーク種別・`experiment_subject`** と整合するよう、スクリプト先頭の定数（`MODEL_PATH`, `IN_CHANNELS`, `USE_NETWORK` 等）を編集して使います。学習時は入力が **0～1** に正規化されるため、通常は **`INPUT_DIV_255 = True`**、RGB 学習なら **`BGR_TO_RGB = True`** を推奨します（コメント内の注意参照）。

## 学習・推論まわりの用語

- **撮像法の組み合わせ** … `use_list` で bf / df / ph の採用を 0/1 で指定。`concatenate` なら有効チャンネルを連結、`alpha` なら比率ブレンド（`blend_particle_size` で刻み）です。
- **`use_list_length`** … 1 / 3 / 9 / 18 で「撮像法のみ」「色チャンネル単位」等の違いを切り替え（制約は `experiment.py` 内の assert 参照）。
- **`compress_rate`** … 保存する予測グレースケールを画素値の刻み幅で量子化（実験クラスの `image_compression_save`）。
