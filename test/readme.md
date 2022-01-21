# test/
  - main.py

  - pic_maker.py

    - 学習用データを作るプログラム

  - lstm_test.py

    - 学習用プログラム

      - lstm_mood_mkdat(username,number,lstm_data,lstm_data_y)

        - mood予測用
        - pic_maker.pyで作った学習用データ（csv形式）を読み込んで、RNN用に整形して返すやつだと思います。

      - lstm_mkdat(username,number,lstm_data,lstm_data_y)
        - emotion予測用
        - pic_maker.pyで作った学習用データ（csv形式）を読み込んで、RNN用に整形して返すやつだと思います。

      - reshape_dat(lstm_data,lstm_data_y)

        - データを、整形し直す。2つめの要素の次元数が`timesteps`になるように。`timesteps`は上の方で定義できます。窓幅みたいなものだと思います。

      - lstm_learn(lstm_data_x,lstm_data_y,data_name)

        - reshape_dat()で作ったデータを使って、学習します。data_nameは保存するモデル（h5のやつ）のファイル名です。

  - only_test.py

    - 学習したモデルを使って予測するだけのプログラム

      - lstm_predict(model_path,lstm_data_x,lstm_data_y)
        - model_pathに多分.h5みたいなモデルを入れて、予測結果の時系列と実測値（答え）の時系列を並べてグラフ描画する
        - lstm_data_x, lstm_data_yはlistだったと思います。

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

