# test/
  - main.py

  - pic_maker.py

    - 学習用データを作るプログラム

  - lstm_test.py

    - 学習用プログラム

  - only_test.py

    - 学習したモデルを使って予測するだけのプログラム

# how to

  - python pic_maker.py

    - 情動データファイルを読み込んで、学習用画像を作る。

  - python3 lstm_test.py

    - 

# archive

  - unit_maker.py
    - 1区切りデータを任意の時系列長に圧縮する。
    - compress arbitrary lentgh data.

  - pca_unit.py
    - 次元圧縮をします。
    - 例えば、6軸IMUのデータ6次元x時系列長ありますが、6次元を2次元とかにします。

